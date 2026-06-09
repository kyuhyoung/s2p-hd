#include "dsm_interpolation.h"

extern "C" {
    void interpolate_with_all_direction(float* dsm_data, int rows_dsm_data, int cols_dsm_data, long long* nan_coords, int rows_nan_coords, int cols_nan_coords)
    {
        int x, y;
        float horizon_value, vertical_value, right_diagonal_value, left_diagonal_value;
        int count;
        float sum;
        for (int ii=0; ii<rows_nan_coords; ii++)
        {
            x = nan_coords[ii * cols_nan_coords+0];
            y = nan_coords[ii * cols_nan_coords+1];
            horizon_value = horizontal_search(x, y, dsm_data, rows_dsm_data, cols_dsm_data);
            vertical_value = vertical_search(x, y, dsm_data, rows_dsm_data, cols_dsm_data);
            right_diagonal_value = right_diagonal_search(x, y, dsm_data, rows_dsm_data, cols_dsm_data);
            left_diagonal_value = left_diagonal_search(x, y, dsm_data, rows_dsm_data, cols_dsm_data);
            
            count = 0;
            sum = 0.;

            if (horizon_value != -9999.)
            {
                count+=1;
                sum+=horizon_value;
            }

            if (vertical_value != -9999.)
            {
                count+=1;
                sum+=vertical_value;
            }

            if (right_diagonal_value != -9999.)
            {
                count+=1;
                sum+=right_diagonal_value;
            }

            if (left_diagonal_value != -9999.)
            {
                count+=1;
                sum+=left_diagonal_value;
            }

            if (count != 0)
            {
                dsm_data[x*cols_dsm_data + y] = sum/float(count);
                // printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\n", x, y, horizon_value, vertical_value, right_diagonal_value, left_diagonal_value, sum/float(count));
            }
        }
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