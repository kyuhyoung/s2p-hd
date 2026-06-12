#pragma once
#include <iostream>
#include <stdio.h>
#include <cmath>

extern "C" {
    using namespace std;

    void interpolate_with_all_direction(float* dsm_data, int rows_dsm_data, int cols_dsm_data, long long* nan_coords, int rows_nan_coords, int cols_nan_coords);
    void interpolate_with_all_direction_mode(float* dsm_data, int rows_dsm_data, int cols_dsm_data, long long* nan_coords, int rows_nan_coords, int cols_nan_coords, int mode);
    float horizontal_search(int x, int y, float* dsm_data, int rows_dsm_data, int cols_dsm_data);
    float vertical_search(int x, int y, float* dsm_data, int rows_dsm_data, int cols_dsm_data);
    float right_diagonal_search(int x, int y, float* dsm_data, int rows_dsm_data, int cols_dsm_data);
    float left_diagonal_search(int x, int y, float* dsm_data, int rows_dsm_data, int cols_dsm_data);
    float select_value(float right_value, float left_value, int right_distance, int left_distance);
    bool possible_area(int up_right_search_x, int up_right_search_y, int rows_dsm_data, int cols_dsm_data);
}