#include "character.h"
#include "array.h"
#include "bits/stdc++.h"

int main(){
    character character_data;
    character_load(character_data, "./resources/character.bin");
    std::cout << character_data.bone_rest_positions(0).x << " " 
                << character_data.bone_rest_positions(0).y <<  " "
                << character_data.bone_rest_positions(0).z  << std::endl;
}