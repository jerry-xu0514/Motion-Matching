#include "character.h"
#include "array.h"
#include "bits/stdc++.h"

int main(){
    character character_data;
    character_load(character_data, "./resources/character.bin");
    std::cout << character_data.positions(1).x << std::endl;
}