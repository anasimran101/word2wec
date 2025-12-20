#pragma once
#include <iostream>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 5
#define MAX_SENTENCE_NUM 6
#define ALIGNMENT_FACTOR 32
#define THREADS_PER_WORD 128
#define BLOCK_SIZE 128

void trainGpu(int sentence_num);
void getResultData();
void initGpu();
void freeGpu();