#include "dsm_interpolation.h"
#include <cstdlib>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// 직렬 의미론 보존 병렬판 (2026-07-06 v2).
//
// 기존 직렬 코드는 무효 픽셀을 래스터 순서로 처리하며 결과를 dsm_data에 즉시
// 써넣어, 뒤 픽셀이 앞서 메운 값을 참조한다(전파). 이 전파가 건물 주변 fill
// 품질(안뜰/그늘 포켓이 지면으로 수렴)을 만들기 때문에, 의미론을 바꾸지 않고
// 병렬화한다. 목표: 직렬과 "비트 동일".
//
// 의존성 분석 (픽셀 (r,c) 처리 시점의 직렬 상태):
//   - 왼쪽/위/대각위 탐색  -> 이미 처리된 픽셀(윗행 전체, 같은 행 왼쪽)의 확정값
//   - 오른쪽/아래/대각아래 -> 아직 처리 안 된 픽셀의 "원본" 상태
// 따라서:
//   - 행을 위에서부터 순차 처리하면 윗행 참조는 모두 확정, 아랫행은 모두 원본.
//   - 같은 행 안에서 run(연속 무효 구간) 내부는 왼쪽 이웃에 의존하므로 순차.
//   - 서로 다른 run은 사이의 유효 픽셀에서 왼쪽 탐색이 멈추므로 독립 -> 병렬.
//   - 오른쪽 탐색만 "행 처리 전" 상태를 봐야 하므로 행 스냅샷(rowbuf)을 읽는다.
// => 결과는 직렬과 비트 동일, 행 내 run들이 병렬로 돈다.
// ---------------------------------------------------------------------------

extern "C" {


    // mode 0: 4방향 평균 (기존 동작)
    // mode 1: 비대칭(ground) fill — 후보 격차가 GROUND_SPREAD_TH 이상이고
    //         "낮은 그룹"(min+TH 이내)이 다수(과반)면 min 채택.
    //
    // v3 (2026-07-06): 직렬 의미론 보존 + 모든 방향탐색 O(1).
    //  - 행을 위에서부터 순차 확정. 같은 행 안의 run(연속 무효 구간)들은 서로
    //    유효 픽셀로 격리되어 독립 -> OpenMP 병렬.
    //  - 위쪽(이미 확정된 행) 탐색: 열/대각선별 "마지막 유효값" 캐시 (행 확정
    //    직후 갱신) -> 직렬의 상향 스캔과 동일한 결과를 O(1)에.
    //  - 아래쪽(아직 원본인 행) 탐색: 열/대각선별 전진 커서(원본만 통과) ->
    //    amortized O(1).
    //  - 행 내부: 오른쪽은 행 스냅샷(rowbuf)의 next-valid 테이블, 왼쪽은 run
    //    시작 경계(항상 유효)에서 출발하는 증분 추적 -> O(1).
    //  결과는 기존 직렬 구현과 비트 동일 (합성/실데이터로 검증).
    void interpolate_with_all_direction_mode(float* dsm_data, int rows_dsm_data, int cols_dsm_data, long long* nan_coords, int rows_nan_coords, int cols_nan_coords, int mode)
    {
        const float GROUND_SPREAD_TH = 5.0f;
        const int R = rows_dsm_data, C = cols_dsm_data;
        const int cc = cols_nan_coords;
        const int ND = R + C - 1;              // 대각선 개수

        float* rowbuf   = (float*)malloc((size_t)C * sizeof(float));
        int*   nxtR     = (int*)  malloc((size_t)C * sizeof(int));    // rowbuf next-valid (>= index)
        float* upV_val  = (float*)malloc((size_t)C * sizeof(float));
        int*   upV_row  = (int*)  malloc((size_t)C * sizeof(int));
        float* upLD_val = (float*)malloc((size_t)ND * sizeof(float)); // d = y - r + (R-1)
        int*   upLD_row = (int*)  malloc((size_t)ND * sizeof(int));
        float* upRD_val = (float*)malloc((size_t)ND * sizeof(float)); // s = y + r
        int*   upRD_row = (int*)  malloc((size_t)ND * sizeof(int));
        int*   dnV_row  = (int*)  malloc((size_t)C * sizeof(int));    // 커서: 다음 원본 유효 (행>rr)
        int*   dnLD_row = (int*)  malloc((size_t)ND * sizeof(int));
        int*   dnRD_row = (int*)  malloc((size_t)ND * sizeof(int));

        if (!rowbuf || !nxtR || !upV_val || !upV_row || !upLD_val || !upLD_row ||
            !upRD_val || !upRD_row || !dnV_row || !dnLD_row || !dnRD_row)
        {
            free(rowbuf); free(nxtR); free(upV_val); free(upV_row);
            free(upLD_val); free(upLD_row); free(upRD_val); free(upRD_row);
            free(dnV_row); free(dnLD_row); free(dnRD_row);
            return;
        }

        for (int y = 0; y < C; y++)  { upV_row[y] = -1; dnV_row[y] = 0; }
        for (int d = 0; d < ND; d++) { upLD_row[d] = -1; upRD_row[d] = -1; dnLD_row[d] = 0; dnRD_row[d] = 0; }

        int scan = 0;
        int row_begin = 0, row_end = 0;

        #pragma omp parallel shared(scan, row_begin, row_end)
        {
            for (int rr = 0; rr < R; rr++)
            {
                #pragma omp single
                {
                    row_begin = scan;
                    while (scan < rows_nan_coords && nan_coords[(size_t)scan * cc] == rr) scan++;
                    row_end = scan;
                    if (row_end > row_begin)
                    {
                        const float* live = dsm_data + (size_t)rr * C;
                        memcpy(rowbuf, live, (size_t)C * sizeof(float));
                        int nx = C;                        // C = 없음
                        for (int y = C - 1; y >= 0; y--)
                        {
                            if (rowbuf[y] != -9999.f) nx = y;
                            nxtR[y] = nx;
                        }
                    }
                }   // barrier

                if (row_end > row_begin)
                {
                    #pragma omp for schedule(dynamic, 64)
                    for (int s0 = row_begin; s0 < row_end; s0++)
                    {
                        const long long yy0 = nan_coords[(size_t)s0 * cc + 1];
                        const bool run_start = (s0 == row_begin) ||
                                               (nan_coords[(size_t)(s0 - 1) * cc + 1] != yy0 - 1);
                        if (!run_start) continue;

                        // run 왼쪽 경계는 정의상 유효 픽셀(또는 행 시작)
                        float lastLv = -9999.f; int lastLy = 0;
                        if (yy0 > 0)
                        {
                            lastLv = dsm_data[(size_t)rr * C + (yy0 - 1)];
                            lastLy = (int)yy0 - 1;
                        }

                        for (int t = s0; t < row_end; t++)
                        {
                            const long long y = nan_coords[(size_t)t * cc + 1];
                            if (t > s0 && y != nan_coords[(size_t)(t - 1) * cc + 1] + 1)
                                break;
                            const int yi = (int)y;

                            float vals[4];

                            // [0] 수평: 오른쪽=rowbuf next-valid, 왼쪽=run 증분 추적
                            {
                                float rv = -9999.f; int rd_ = 1;
                                if (yi + 1 <= C - 1)
                                {
                                    const int k = nxtR[yi + 1];
                                    if (k < C) { rv = rowbuf[k]; rd_ = k - yi; }
                                }
                                if (rv != -9999.f && lastLv != -9999.f)
                                    vals[0] = select_value(rv, lastLv, rd_, yi - lastLy);
                                else vals[0] = -9999.f;
                            }

                            // [1] 수직: 위=upV 캐시, 아래=dnV 커서(원본)
                            {
                                float uv = -9999.f; int ud = 1;
                                if (upV_row[yi] >= 0) { uv = upV_val[yi]; ud = rr - upV_row[yi]; }
                                float dv = -9999.f; int dd = 1;
                                int r2 = dnV_row[yi]; if (r2 <= rr) r2 = rr + 1;
                                while (r2 < R && dsm_data[(size_t)r2 * C + yi] == -9999.f) r2++;
                                dnV_row[yi] = r2;
                                if (r2 <= R - 1) { dv = dsm_data[(size_t)r2 * C + yi]; dd = r2 - rr; }
                                if (uv != -9999.f && dv != -9999.f)
                                    vals[1] = select_value(uv, dv, ud, dd);
                                else vals[1] = -9999.f;
                            }

                            const bool border = (rr == 0 || rr == R - 1 || yi == 0 || yi == C - 1);

                            // [2] 대각 ↗(위) / ↙(아래) : s = y + r
                            if (border) vals[2] = -9999.f;
                            else
                            {
                                const int s = yi + rr;
                                float uv = -9999.f; int ud = 1;
                                if (upRD_row[s] >= 0)
                                {
                                    const int r1 = upRD_row[s];
                                    uv = upRD_val[s]; ud = 2 * (rr - r1);
                                }
                                float dv = -9999.f; int dd = 1;
                                const int rmax = (R - 1 < s) ? (R - 1) : s;   // y=s-r >= 0
                                int r2 = dnRD_row[s]; if (r2 <= rr) r2 = rr + 1;
                                while (r2 <= rmax && dsm_data[(size_t)r2 * C + (s - r2)] == -9999.f) r2++;
                                dnRD_row[s] = r2;
                                if (r2 <= rmax) { dv = dsm_data[(size_t)r2 * C + (s - r2)]; dd = 2 * (r2 - rr); }
                                if (uv != -9999.f && dv != -9999.f)
                                    vals[2] = select_value(uv, dv, ud, dd);
                                else vals[2] = -9999.f;
                            }

                            // [3] 대각 ↖(위) / ↘(아래) : d = y - r + (R-1)
                            if (border) vals[3] = -9999.f;
                            else
                            {
                                const int d = yi - rr + (R - 1);
                                float uv = -9999.f; int ud = 1;
                                if (upLD_row[d] >= 0)
                                {
                                    const int r1 = upLD_row[d];
                                    uv = upLD_val[d]; ud = 2 * (rr - r1);
                                }
                                float dv = -9999.f; int dd = 1;
                                const int rmax_c = (C - 1) - yi + rr;          // y=yi+(r-rr) <= C-1
                                const int rmax = (R - 1 < rmax_c) ? (R - 1) : rmax_c;
                                int r2 = dnLD_row[d]; if (r2 <= rr) r2 = rr + 1;
                                while (r2 <= rmax && dsm_data[(size_t)r2 * C + (yi + (r2 - rr))] == -9999.f) r2++;
                                dnLD_row[d] = r2;
                                if (r2 <= rmax) { dv = dsm_data[(size_t)r2 * C + (yi + (r2 - rr))]; dd = 2 * (r2 - rr); }
                                if (uv != -9999.f && dv != -9999.f)
                                    vals[3] = select_value(uv, dv, ud, dd);
                                else vals[3] = -9999.f;
                            }

                            int count = 0, low_count;
                            float sum = 0.f, vmin = 0.f, vmax = 0.f;
                            for (int k = 0; k < 4; k++)
                            {
                                if (vals[k] != -9999.f)
                                {
                                    if (count == 0) { vmin = vals[k]; vmax = vals[k]; }
                                    else
                                    {
                                        if (vals[k] < vmin) vmin = vals[k];
                                        if (vals[k] > vmax) vmax = vals[k];
                                    }
                                    count += 1;
                                    sum += vals[k];
                                }
                            }

                            if (count != 0)
                            {
                                bool use_min = false;
                                if (mode == 1 && (vmax - vmin) >= GROUND_SPREAD_TH)
                                {
                                    low_count = 0;
                                    for (int k = 0; k < 4; k++)
                                        if (vals[k] != -9999.f && vals[k] <= vmin + GROUND_SPREAD_TH)
                                            low_count += 1;
                                    if (2 * low_count >= count) use_min = true;
                                }

                                const float outv = use_min ? vmin : (sum / float(count));
                                dsm_data[(size_t)rr * C + yi] = outv;
                                lastLv = outv; lastLy = yi;
                            }
                            // count==0 이면 -9999 유지, 왼쪽 추적도 그대로 (직렬과 동일)
                        }
                    }   // barrier: 이 행 fill 확정
                }

                // 행 확정 후 위쪽 캐시 갱신 (유효값이 하나라도 있는 행만)
                #pragma omp for schedule(static)
                for (int y = 0; y < C; y++)
                {
                    const float v = dsm_data[(size_t)rr * C + y];
                    if (v != -9999.f)
                    {
                        upV_val[y] = v;  upV_row[y] = rr;
                        const int d = y - rr + (R - 1);
                        upLD_val[d] = v; upLD_row[d] = rr;
                        const int s = y + rr;
                        upRD_val[s] = v; upRD_row[s] = rr;
                    }
                }   // barrier
            }
        }

        free(rowbuf); free(nxtR); free(upV_val); free(upV_row);
        free(upLD_val); free(upLD_row); free(upRD_val); free(upRD_row);
        free(dnV_row); free(dnLD_row); free(dnRD_row);
    }

    // 기존 ABI 유지 — 평균 모드로 위임
    void interpolate_with_all_direction(float* dsm_data, int rows_dsm_data, int cols_dsm_data, long long* nan_coords, int rows_nan_coords, int cols_nan_coords)
    {
        interpolate_with_all_direction_mode(dsm_data, rows_dsm_data, cols_dsm_data, nan_coords, rows_nan_coords, cols_nan_coords, 0);
    }

    float horizontal_search(int x, int y, float* dsm_data, int rows_dsm_data, int cols_dsm_data)
    {
        int start_y = y;
        int right_search = start_y;
        int left_search = start_y;
        float right_value = -9999.;
        float left_value = -9999.;
        int right_distance = 1;
        int left_distance = 1;

        while (right_search < cols_dsm_data-1)
        {
            if (dsm_data[x*cols_dsm_data + (right_search+1)] == -9999.)
            {
                right_search+=1;
            }
            else
            {
                right_value = dsm_data[x*cols_dsm_data + (right_search+1)];
                right_distance = (right_search+1) - start_y;
                break;
            }
        }

        while (left_search > 0)
        {
            if (dsm_data[x*cols_dsm_data + (left_search-1)] == -9999.)
            {
                left_search-=1;
            }
            else
            {
                left_value = dsm_data[x*cols_dsm_data + (left_search-1)];
                left_distance = start_y - (left_search-1);
                break;
            }
        }

        if ((right_value != -9999.) and (left_value != -9999.))
        {
            return select_value(right_value, left_value, right_distance, left_distance);
        }

        return -9999.;
    }

    float vertical_search(int x, int y, float* dsm_data, int rows_dsm_data, int cols_dsm_data)
    {
        int start_x = x;
        int up_search = start_x;
        int down_search = start_x;
        float up_value = -9999.;
        float down_value = -9999.;
        int up_distance = 1;
        int down_distance = 1;

        while (up_search > 0)
        {
            if (dsm_data[(up_search-1)*cols_dsm_data + (y)] == -9999.)
            {
                up_search-=1;
            }
            else
            {
                up_value = dsm_data[(up_search-1)*cols_dsm_data + (y)];
                up_distance = start_x - (up_search-1);
                break;
            }
        }

        while (down_search < (rows_dsm_data-1))
        {
            if (dsm_data[(down_search+1)*cols_dsm_data + (y)] == -9999.)
            {
                down_search+=1;
            }
            else
            {
                down_value = dsm_data[(down_search+1)*cols_dsm_data + (y)];
                down_distance = (down_search+1) - start_x;
                break;
            }
        }

        if ((up_value != -9999.) and (down_value != -9999.))
        {
            return select_value(up_value, down_value, up_distance, down_distance);
        }

        return -9999.;
    }

    float right_diagonal_search(int x, int y, float* dsm_data, int rows_dsm_data, int cols_dsm_data)
    {
        int start_x = x;
        int start_y = y;
        int up_right_search_x = start_x;
        int up_right_search_y = start_y;
        int down_left_search_x = start_x;
        int down_left_search_y = start_y;
        float up_right_value = -9999.;
        float down_left_value = -9999.;
        int up_right_distance = 1;
        int down_left_distance = 1;

        while (possible_area(up_right_search_x, up_right_search_y, rows_dsm_data, cols_dsm_data))
        {
            if (dsm_data[(up_right_search_x-1)*cols_dsm_data + (up_right_search_y+1)] == -9999.)
            {
                up_right_search_x -= 1;
                up_right_search_y += 1;
            }
            else
            {
                up_right_value = dsm_data[(up_right_search_x-1)*cols_dsm_data + (up_right_search_y+1)];
                up_right_distance = start_x - (up_right_search_x-1) + (up_right_search_y+1) - start_y;
                break;
            }
        }

        while (possible_area(down_left_search_x, down_left_search_y, rows_dsm_data, cols_dsm_data))
        {
            if (dsm_data[(down_left_search_x+1)*cols_dsm_data + (down_left_search_y-1)] == -9999.)
            {
                down_left_search_x += 1;
                down_left_search_y -= 1;
            }
            else
            {
                down_left_value = dsm_data[(down_left_search_x+1)*cols_dsm_data + (down_left_search_y-1)];
                down_left_distance = (down_left_search_x+1) - start_x + start_y - (down_left_search_y-1);
                break;
            }
        }

        if ((up_right_value != -9999.) and (down_left_value != -9999.))
        {
            return select_value(up_right_value, down_left_value, up_right_distance, down_left_distance);
        }

        return -9999.;
    }

    float left_diagonal_search(int x, int y, float* dsm_data, int rows_dsm_data, int cols_dsm_data)
    {
        int start_x = x;
        int start_y = y;
        int up_left_search_x = start_x;
        int up_left_search_y = start_y;
        int down_right_search_x = start_x;
        int down_right_search_y = start_y;
        float up_left_value = -9999.;
        float down_right_value = -9999.;
        int up_left_distance = 1;
        int down_right_distance = 1;

        while (possible_area(up_left_search_x, up_left_search_y, rows_dsm_data, cols_dsm_data))
        {
            if (dsm_data[(up_left_search_x-1)*cols_dsm_data + (up_left_search_y-1)] == -9999.)
            {
                up_left_search_x -= 1;
                up_left_search_y -= 1;
            }
            else
            {
                up_left_value = dsm_data[(up_left_search_x-1)*cols_dsm_data + (up_left_search_y-1)];
                up_left_distance = start_x - (up_left_search_x-1) + start_y - (up_left_search_y-1);
                break;
            }
        }

        while (possible_area(down_right_search_x, down_right_search_y, rows_dsm_data, cols_dsm_data))
        {
            if (dsm_data[(down_right_search_x+1)*cols_dsm_data + (down_right_search_y+1)] == -9999.)
            {
                down_right_search_x += 1;
                down_right_search_y += 1;
            }
            else
            {
                down_right_value = dsm_data[(down_right_search_x+1)*cols_dsm_data + (down_right_search_y+1)];
                down_right_distance = (down_right_search_x+1) - start_x + (down_right_search_y+1) - start_y;
                break;
            }
        }

        if ((up_left_value != -9999.) and (down_right_value != -9999.))
        {
            return select_value(up_left_value, down_right_value, up_left_distance, down_right_distance);
        }

        return -9999.;
    }

    float select_value(float right_value, float left_value, int right_distance, int left_distance)
    {
        int total_distance = right_distance + left_distance;
        float point;

        if (right_value < left_value) point = right_distance / float(total_distance);
        else point = left_distance / float(total_distance);

        float index = log2(1/(1-point));

        int weights;
        if (int(index) >= 50) weights = 999;
        else weights = int(index)+1;

        return min(right_value, left_value) + (weights / 1000.) * abs(right_value - left_value);
    }

    bool possible_area(int x, int y, int rows_dsm_data, int cols_dsm_data)
    {
        if (((x > 0) and (x < (rows_dsm_data - 1))) and ((y > 0) and (y < (cols_dsm_data - 1)))) return true;
        return false;
    }


}
