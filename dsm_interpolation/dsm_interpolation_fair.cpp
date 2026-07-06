#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// 공평(순서무관) fill 실험판 (2026-07-06).
//
// 원리:
//  - 스냅샷 의미론: 모든 탐색은 "메우기 전" 원본만 참조 -> 처리 순서/스레드와
//    무관하게 결정적. (직렬판의 좌상->우하 cascade 편향 제거)
//  - 8개 라인 패밀리(수평/수직/대각2/기울기 1:2, 2:1 4종)로 픽셀당 최대
//    16개의 실측 끝점 샘플(값, 유클리드 거리)을 모은다. 라인은 양끝이 모두
//    있어야 채택(보간만 허용, 외삽 금지 — 원본과 동일한 fillable 기준).
//  - 규칙(ground prior의 명시화):
//      샘플 격차(vmax-vmin) < TH(5m): 순수 지형/옥상 내부 구멍 -> 전체 샘플
//          거리가중(IDW) 보간 (공평, 경사 유지)
//      격차 >= TH: 지붕+지면이 섞인 occlusion 구멍 -> "낮은 그룹"(vmin+TH 이내)
//          샘플만으로 IDW (가려진 곳은 지면이라는 prior를 순서가 아니라
//          규칙으로 명시)
//  - mode 0: low-group 규칙 없이 전체 IDW (참고용 공평-avg)
//    mode 1: 위의 low-group 규칙 적용 (제안 기본)
//
// ABI는 기존과 동일한 interpolate_with_all_direction_mode 를 노출한다.
// ---------------------------------------------------------------------------

extern "C" {

static const float NODATA = -9999.f;
static const float SPREAD_TH = 5.0f;

// ---- v4: 구멍 connected-component(4-연결) 라벨링 + 성분 경계 조성 판정 ----
// 판정의 근거를 "픽셀에서 보이는 광선"이 아니라 "구멍 성분에 인접한 실측 경계"로
// 옮긴다. 지붕 속 구멍은 경계가 전부 지붕(단봉) -> 지붕으로만 보간.
// 지붕+지면이 섞인 경계(이봉) = occlusion -> 성분 전체를 낮은 경계 그룹으로.
// (v1~v3의 per-pixel 광선 규칙은 나이트스텝 누수/림 흔들림 문제가 있었음)

static std::vector<unsigned char> g_bimodal;
static std::vector<float> g_lowMean;

static int uf_find(std::vector<int>& P, int a){ while(P[a]!=a){ P[a]=P[P[a]]; a=P[a]; } return a; }
static void uf_union(std::vector<int>& P, int a, int b){ a=uf_find(P,a); b=uf_find(P,b); if(a<b) P[b]=a; else if(b<a) P[a]=b; }

struct Dir { int dx, dy; float step; };

static void sweep_fair(const float* dsm, int R, int C,
                       long long x0, long long y0, int dx, int dy, float step,
                       const float* cut_p,
                       float* num_p, float* den_p,
                       float* bv, int* bd)
{
    long long len = 0;
    {
        long long x = x0, y = y0;
        while (x >= 0 && x < R && y >= 0 && y < C) { len++; x += dx; y += dy; }
    }
    {
        float lastv = NODATA; long long lasti = 0;
        long long x = x0, y = y0;
        for (long long i = 0; i < len; i++, x += dx, y += dy)
        {
            const float v = dsm[(size_t)x * C + y];
            if (v != NODATA) { lastv = v; lasti = i; }
            bv[i] = lastv; bd[i] = (int)(i - lasti);
        }
    }
    {
        float av = NODATA; long long ai = 0;
        long long x = x0 + (len - 1) * dx, y = y0 + (len - 1) * dy;
        for (long long i = len - 1; i >= 0; i--, x -= dx, y -= dy)
        {
            const float v = dsm[(size_t)x * C + y];
            if (v != NODATA) { av = v; ai = i; continue; }
            const float behind = bv[i];
            if (behind == NODATA || av == NODATA) continue;
            const size_t idx = (size_t)x * C + y;
            const float d_b = step * (float)bd[i];
            const float d_a = step * (float)(ai - i);
            const float cut = cut_p[idx];
            if (behind <= cut) { num_p[idx] += behind / d_b; den_p[idx] += 1.f / d_b; }
            if (av <= cut)     { num_p[idx] += av / d_a;     den_p[idx] += 1.f / d_a; }
        }
    }
}

// 패밀리의 모든 라인 시작점 나열: 전임자 (x-dx, y-dy) 가 격자 밖인 셀
static void family_starts(int R, int C, int dx, int dy,
                          std::vector<long long>& xs, std::vector<long long>& ys)
{
    xs.clear(); ys.clear();
    for (long long x = 0; x < R; x++)
        for (int k = 0; k < dy; k++)   // y-dy < 0  =>  y in [0, dy)
        { if (k < C) { xs.push_back(x); ys.push_back(k); } }
    if (dy == 0)
        for (long long y = 0; y < C; y++) { xs.push_back(0); ys.push_back(y); }
    else
        for (long long y = dy; y < C; y++)   // x-dx < 0 (x in [0,dx)) 이고 y-dy >= 0
            for (int k = 0; k < dx; k++)
                if (k < R) { xs.push_back(k); ys.push_back(y); }
}

void interpolate_with_all_direction_mode(float* dsm_data, int rows_dsm_data, int cols_dsm_data, long long* nan_coords, int rows_nan_coords, int cols_nan_coords, int mode)
{
    const int R = rows_dsm_data, C = cols_dsm_data;
    const int cc = cols_nan_coords;
    const size_t N = (size_t)R * C;
    const float SQ2 = sqrtf(2.f), SQ5 = sqrtf(5.f);
    const Dir dirs[8] = {
        {0, 1, 1.f}, {1, 0, 1.f}, {1, 1, SQ2}, {1, -1, SQ2},
        {1, 2, SQ5}, {2, 1, SQ5}, {1, -2, SQ5}, {2, -1, SQ5},
    };

    float* num_p = (float*)malloc(N * sizeof(float));
    float* den_p = (float*)malloc(N * sizeof(float));
    float* cut_p = (float*)malloc(N * sizeof(float));
    int*   lab_p = (int*)  malloc(N * sizeof(int));
    signed char* side_p = (signed char*)malloc(N);   // 0=미귀속/단봉, 1=low측, 2=high측
    if (!num_p || !den_p || !cut_p || !lab_p || !side_p)
    { free(num_p); free(den_p); free(cut_p); free(lab_p); free(side_p); return; }

    #pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < rows_nan_coords; ii++)
    {
        const size_t idx = (size_t)nan_coords[(size_t)ii * cc] * C + nan_coords[(size_t)ii * cc + 1];
        num_p[idx] = 0.f; den_p[idx] = 0.f; cut_p[idx] = 1e30f; side_p[idx] = 0;
    }

    // ---- (A) 4-연결 성분 라벨링 (run 기반 union-find; coords는 래스터 정렬) ----
    std::vector<int> parent;
    {
        struct Run { long long y0, y1; int lab; };
        std::vector<Run> prev, cur;
        long long ii = 0;
        for (int rr = 0; rr < R && ii < rows_nan_coords; rr++)
        {
            cur.clear();
            if (nan_coords[(size_t)ii * cc] != rr) { prev.swap(cur); continue; }
            while (ii < rows_nan_coords && nan_coords[(size_t)ii * cc] == rr)
            {
                long long y0 = nan_coords[(size_t)ii * cc + 1], y1 = y0; ii++;
                while (ii < rows_nan_coords && nan_coords[(size_t)ii * cc] == rr &&
                       nan_coords[(size_t)ii * cc + 1] == y1 + 1) { y1++; ii++; }
                int lab = -1;
                for (const Run& pr : prev)
                {
                    if (pr.y1 < y0) continue;
                    if (pr.y0 > y1) break;
                    if (lab < 0) lab = uf_find(parent, pr.lab); else uf_union(parent, lab, pr.lab);
                }
                if (lab < 0) { lab = (int)parent.size(); parent.push_back(lab); }
                cur.push_back({y0, y1, lab});
            }
            for (const Run& cr : cur)
                for (long long y = cr.y0; y <= cr.y1; y++) lab_p[(size_t)rr * C + y] = cr.lab;
            prev.swap(cur);
        }
    }
    const int NL = (int)parent.size();
    std::vector<int> root(NL);
    for (int i = 0; i < NL; i++) root[i] = uf_find(parent, i);

    // ---- (B) 성분 경계 min/max (8-이웃 실측) ----
    std::vector<float> cmin(NL, 1e30f), cmax(NL, -1e30f);
    #pragma omp parallel
    {
        std::vector<float> lmin(NL, 1e30f), lmax(NL, -1e30f);
        #pragma omp for schedule(static) nowait
        for (long long k = 0; k < rows_nan_coords; k++)
        {
            const long long x = nan_coords[(size_t)k * cc], y = nan_coords[(size_t)k * cc + 1];
            const int lb = root[lab_p[(size_t)x * C + y]];
            for (int dx8 = -1; dx8 <= 1; dx8++) for (int dy8 = -1; dy8 <= 1; dy8++)
            {
                if (!dx8 && !dy8) continue;
                const long long xx = x + dx8, yy = y + dy8;
                if (xx < 0 || xx >= R || yy < 0 || yy >= C) continue;
                const float v = dsm_data[(size_t)xx * C + yy];
                if (v == NODATA) continue;
                if (v < lmin[lb]) lmin[lb] = v;
                if (v > lmax[lb]) lmax[lb] = v;
            }
        }
        #pragma omp critical
        for (int i = 0; i < NL; i++)
        {
            if (lmin[i] < cmin[i]) cmin[i] = lmin[i];
            if (lmax[i] > cmax[i]) cmax[i] = lmax[i];
        }
    }

    // ---- (C) 후보 성분(range>=2TH)만 64-bin 히스토그램 -> Otsu 임계 ----
    std::vector<int> cand(NL, -1);
    int ncand = 0;
    for (int i = 0; i < NL; i++)
        if (root[i] == i && cmax[i] - cmin[i] >= 2 * SPREAD_TH) cand[i] = ncand++;
    const int NB = 64;
    std::vector<long long> hist((size_t)ncand * NB, 0);
    std::vector<double> hsum((size_t)ncand * NB, 0.0);
    #pragma omp parallel for schedule(static)
    for (long long k = 0; k < rows_nan_coords; k++)
    {
        const long long x = nan_coords[(size_t)k * cc], y = nan_coords[(size_t)k * cc + 1];
        const int lb = root[lab_p[(size_t)x * C + y]];
        const int ci = cand[lb];
        if (ci < 0) continue;
        const float lo = cmin[lb], span = cmax[lb] - cmin[lb];
        for (int dx8 = -1; dx8 <= 1; dx8++) for (int dy8 = -1; dy8 <= 1; dy8++)
        {
            if (!dx8 && !dy8) continue;
            const long long xx = x + dx8, yy = y + dy8;
            if (xx < 0 || xx >= R || yy < 0 || yy >= C) continue;
            const float v = dsm_data[(size_t)xx * C + yy];
            if (v == NODATA) continue;
            int b = (int)((v - lo) / span * (NB - 1) + 0.5f);
            if (b < 0) b = 0; if (b > NB - 1) b = NB - 1;
            #pragma omp atomic
            hist[(size_t)ci * NB + b]++;
            #pragma omp atomic
            hsum[(size_t)ci * NB + b] += (double)v;
        }
    }
    // Otsu per candidate + 판정: 군집 평균 분리 >= 2TH & 양군 >= 3%
    std::vector<float> thr(NL, 1e30f);       // 성분별 low 컷 (bimodal일 때만 finite)
    std::vector<float> lowMean(NL, NODATA);
    for (int i = 0; i < NL; i++)
    {
        const int ci = cand[i];
        if (ci < 0) continue;
        const long long* h = &hist[(size_t)ci * NB];
        const double* hs = &hsum[(size_t)ci * NB];
        long long tot = 0; double stot = 0;
        for (int b = 0; b < NB; b++) { tot += h[b]; stot += hs[b]; }
        if (tot < 8) continue;
        // 전체 분산 (히스토그램 근사) — Otsu 품질비의 분모
        double mu_all = stot / (double)tot, var_all = 0;
        {
            const float lo2 = cmin[i], span2 = cmax[i] - cmin[i];
            for (int b = 0; b < NB; b++)
            {
                if (!h[b]) continue;
                const double vb = lo2 + span2 * (b + 0.5) / (NB - 1);
                var_all += (double)h[b] * (vb - mu_all) * (vb - mu_all);
            }
        }
        long long w0 = 0; double s0 = 0; double best = -1; int bestb = -1;
        double bmu0 = 0, bmu1 = 0; long long bw0 = 0;
        for (int b = 0; b < NB - 1; b++)
        {
            w0 += h[b]; s0 += hs[b];
            const long long w1 = tot - w0;
            if (w0 == 0 || w1 == 0) continue;
            const double mu0 = s0 / w0, mu1 = (stot - s0) / w1;
            const double bc = (double)w0 * (double)w1 / (double)tot * (mu0 - mu1) * (mu0 - mu1);
            if (bc > best) { best = bc; bestb = b; bmu0 = mu0; bmu1 = mu1; bw0 = w0; }
        }
        if (bestb < 0 || var_all <= 0) continue;
        const long long w1 = tot - bw0;
        // 이봉 판정: 군집분리 >= 2TH, 양군 >= 2샘플(측지 귀속이 오탐 피해를 국소화),
        // Otsu 품질비 >= 0.85 (연속 경사의 상한 ~0.75 -> 원리적으로 배제)
        if (bmu1 - bmu0 >= 2 * SPREAD_TH && bw0 >= 2 && w1 >= 2 &&
            best / var_all >= 0.85)
        {
            thr[i] = (float)(cmin[i] + (cmax[i] - cmin[i]) * (bestb + 0.5f) / (NB - 1));
            lowMean[i] = (float)bmu0;
        }
    }

    // ---- (D) 이봉 성분 내부 측지 귀속: low/high 경계에서 동시 BFS (low 우선) ----
    {
        std::vector<size_t> qlow, qhigh, nlow, nhigh;
        // 시드: 이봉 성분의 구멍 픽셀 중 경계 실측과 인접한 것
        for (long long k = 0; k < rows_nan_coords; k++)
        {
            const long long x = nan_coords[(size_t)k * cc], y = nan_coords[(size_t)k * cc + 1];
            const int lb = root[lab_p[(size_t)x * C + y]];
            if (thr[lb] >= 1e29f) continue;
            const size_t idx = (size_t)x * C + y;
            bool nearLow = false, nearHigh = false;
            for (int dx8 = -1; dx8 <= 1; dx8++) for (int dy8 = -1; dy8 <= 1; dy8++)
            {
                if (!dx8 && !dy8) continue;
                const long long xx = x + dx8, yy = y + dy8;
                if (xx < 0 || xx >= R || yy < 0 || yy >= C) continue;
                const float v = dsm_data[(size_t)xx * C + yy];
                if (v == NODATA) continue;
                if (v <= thr[lb]) nearLow = true; else nearHigh = true;
            }
            if (nearLow)       { side_p[idx] = 1; qlow.push_back(idx); }
            else if (nearHigh) { side_p[idx] = 2; qhigh.push_back(idx); }
        }
        // 레벨 동기 BFS (4-연결, low 전선이 동레벨 경합에서 승리 -> 결정적)
        const long long NX[4] = {-1, 1, 0, 0};
        const long long NY[4] = {0, 0, -1, 1};
        while (!qlow.empty() || !qhigh.empty())
        {
            nlow.clear(); nhigh.clear();
            for (size_t idx : qlow)
            {
                const long long x = (long long)(idx / C), y = (long long)(idx % C);
                for (int d = 0; d < 4; d++)
                {
                    const long long xx = x + NX[d], yy = y + NY[d];
                    if (xx < 0 || xx >= R || yy < 0 || yy >= C) continue;
                    const size_t j = (size_t)xx * C + yy;
                    if (dsm_data[j] != NODATA || side_p[j] != 0) continue;
                    side_p[j] = 1; nlow.push_back(j);
                }
            }
            for (size_t idx : qhigh)
            {
                const long long x = (long long)(idx / C), y = (long long)(idx % C);
                for (int d = 0; d < 4; d++)
                {
                    const long long xx = x + NX[d], yy = y + NY[d];
                    if (xx < 0 || xx >= R || yy < 0 || yy >= C) continue;
                    const size_t j = (size_t)xx * C + yy;
                    if (dsm_data[j] != NODATA || side_p[j] != 0) continue;
                    side_p[j] = 2; nhigh.push_back(j);
                }
            }
            qlow.swap(nlow); qhigh.swap(nhigh);
        }
    }

    // ---- (E) 픽셀 컷 확정: low측 -> thr, 그 외 -> +inf ----
    #pragma omp parallel for schedule(static)
    for (long long k = 0; k < rows_nan_coords; k++)
    {
        const long long x = nan_coords[(size_t)k * cc], y = nan_coords[(size_t)k * cc + 1];
        const size_t idx = (size_t)x * C + y;
        const int lb = root[lab_p[(size_t)x * C + y]];
        lab_p[idx] = lb;
        if (mode == 1 && side_p[idx] == 1) cut_p[idx] = thr[lb];
    }

    // ---- (F) 8패밀리 스윕: cut 조건부 IDW ----
    const long long maxlen = (long long)R + C;
    std::vector<long long> xs, ys;
    for (int f = 0; f < 8; f++)
    {
        const Dir& D = dirs[f];
        if (D.dy >= 0) family_starts(R, C, D.dx, D.dy, xs, ys);
        else
        {
            xs.clear(); ys.clear();
            const int ady = -D.dy;
            for (long long x = 0; x < R; x++)
                for (int k = 0; k < ady; k++)
                    if (C - 1 - k >= 0) { xs.push_back(x); ys.push_back(C - 1 - k); }
            for (long long y = 0; y + ady < C; y++)
                for (int k = 0; k < D.dx; k++)
                    if (k < R) { xs.push_back(k); ys.push_back(y); }
        }
        const long long nl = (long long)xs.size();
        #pragma omp parallel
        {
            std::vector<float> bv(maxlen);
            std::vector<int> bd(maxlen);
            #pragma omp for schedule(dynamic, 16)
            for (long long li = 0; li < nl; li++)
                sweep_fair(dsm_data, R, C, xs[li], ys[li], D.dx, D.dy, D.step,
                           cut_p, num_p, den_p, bv.data(), bd.data());
        }
    }

    // ---- (G) 커밋 (+ low측 사각지대는 성분 low 군집 평균으로) ----
    #pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < rows_nan_coords; ii++)
    {
        const size_t idx = (size_t)nan_coords[(size_t)ii * cc] * C + nan_coords[(size_t)ii * cc + 1];
        if (den_p[idx] > 0.f) dsm_data[idx] = num_p[idx] / den_p[idx];
        else if (mode == 1 && side_p[idx] == 1)
        {
            const float lm = lowMean[lab_p[idx]];
            if (lm != NODATA) dsm_data[idx] = lm;
        }
    }

    free(num_p); free(den_p); free(cut_p); free(lab_p); free(side_p);
}

void interpolate_with_all_direction(float* dsm_data, int rows_dsm_data, int cols_dsm_data, long long* nan_coords, int rows_nan_coords, int cols_nan_coords)
{
    interpolate_with_all_direction_mode(dsm_data, rows_dsm_data, cols_dsm_data, nan_coords, rows_nan_coords, cols_nan_coords, 1);
}

}
