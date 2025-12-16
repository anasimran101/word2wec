#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <vocab.h>
#include <vector>
#include <unordered_map>
#include <map>
#include <skipgram.h>

#define MAX_SENTENCE_LENGTH 10
#define MAX_SENTENCE_NUM 10

std::unordered_map<std::string, int> vocab_hash;
std::vector<word_index> vocab_list;

std::string train_corpus_file, output_file;
std::string save_vocab_file, read_vocab_file;
struct word_index *vocab;
std::ifstream fin;

int window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

size_t vocab_size = 0, layer1_size = 100, layer1_size_aligned;

long long train_words = 0, word_count_actual = 0, file_size = 0;
int epochs = 5;
double alpha = 0.025, starting_alpha, sample = 1e-3;
double *syn0;
int *sen;
clock_t start;
size_t table_size = 100;
int *table;
bool binary = false;

// calculate wj^0.75 / SUM i= 0->n (w_i^0.75)
void initUnigramDistribuiton()
{
    table = new int[table_size];
    double a = 0.75;
    double sum_p = 0.0, p_w;
    size_t index = 0, prev_index = 0;
    for (size_t i = 0; i < vocab_list.size(); i++)
        sum_p += pow(vocab_list[i].count, a);
    for (size_t i = 0; i < vocab_list.size(); i++)
    {
        p_w += pow(vocab_list[i].count, a) / sum_p;
        prev_index = index;
        index = p_w * table_size;
        std::cout << "Word: " << vocab_list[i].word_index << " Prob: " << p_w << " Index: " << index << std::endl;
        for (size_t j = prev_index; j < index && j < table_size; j++)
        {
            table[j] = i;
        }
    }
    while (index < table_size)
        table[index++] = vocab_list.size() - 1;
}

void InitNet();


void *TrainModelThread(void *id)
{
    int word_index, sentence_length = 0;
    long long word_count = 0, last_word_count = 0;
    int local_iter = epochs; //check
    unsigned int next_random = (long)id;
    int sentence_num;
    clock_t now;
    float *alpha_ptr = (float *)sen + MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH;
    std::ifstream fi(train_corpus_file, std::ios::in);
    fi.seekg(std::ios::pos_type(file_size / (int)num_threads * (long)id));
    sentence_length = 0;
    sentence_num = 0;
    while (1)
    {
        if (word_count - last_word_count > 10000)
        {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            now = clock();
            printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                    word_count_actual / (float)(epochs * train_words + 1) * 100,
                    word_count_actual / ((float)(now - start + 1) / (float)CLOCKS_PER_SEC * 1000));
            std::cout.flush();
            alpha = starting_alpha * (1 - word_count_actual / (float)(epochs * train_words + 1));
            if (alpha < starting_alpha * 0.0001)
                alpha = starting_alpha * 0.0001;
        }
        std::string word;
        int word_index;
        while (1)
        {
            loadWordFromFile(word, fi);
            word_index = getWordIndex(word); 
            if (fi.eof())
                break;
            if (word_index == -1)
                continue;
            word_count++;
            if (word_index == 0)
                break;
            // The subsampling randomly discards frequent words while keeping the ranking same
            if (sample > 0)
            {
                float ran = (sqrt(vocab_list[word_index].count / (sample * train_words)) + 1) * (sample * train_words) / vocab_list[word_index].count;
                next_random = next_random * (unsigned int)1664525 + 1013904223;
                if (ran < (next_random & 0xFFFF) / (float)65536)
                    continue;
            }
            sen[sentence_num * MAX_SENTENCE_LENGTH + sentence_length] = word_index;
            sentence_length++;
            if (sentence_length >= MAX_SENTENCE_LENGTH)
            {
                alpha_ptr[sentence_num] = alpha;
                sentence_num++;
                sentence_length = 0;
                if (sentence_num >= MAX_SENTENCE_NUM)
                    break;
            }
        }

        // Do GPU training here
        trainGpu(sentence_num);
        //////////////////////
        sentence_num = 0;
        sentence_length = 0;

        if (fi.eof() || (word_count > train_words / num_threads))
        {
            word_count_actual += word_count - last_word_count;
            local_iter--;
            if (local_iter == 0)
                break;
            word_count = 0;
            last_word_count = 0;
            fi.seekg(std::ios::pos_type(file_size / (int)num_threads * (long)id));
        }
    }
    getResultData();
    fi.close();
    pthread_exit(NULL);
}

void TrainModel()
{
    long a, b, c, d;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    std::cout << "Starting training using file " << train_corpus_file << std::endl;
    starting_alpha = alpha;
    if (!read_vocab_file.empty())
        loadFromVocabFile(read_vocab_file);
    else
        loadFromTrainFile(train_corpus_file);
    if (!save_vocab_file.empty())
        saveVocab();
    if (output_file[0] == 0)
        return;
    InitNet();
    
    initUnigramDistribuiton();
    initGpu();
    start = clock();
    for (a = 0; a < num_threads; a++)
        pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++)
        pthread_join(pt[a], NULL);
    freeGpu();
    // Save the word_index vectors

    std::ofstream out(output_file, binary ? std::ios::binary : std::ios::out);
    

    out << vocab_size << " " << layer1_size << std::endl;
    for (size_t i = 0; i < vocab_size; i++)
    {
        out >> vocab_list[i] << " ";
        if (binary)
            for (b = 0; b < layer1_size; b++)
                out << syn0[i * layer1_size_aligned + b] << " ";
        else
            for (b = 0; b < layer1_size; b++)
                out << syn0[i * layer1_size_aligned + b] << " ";
        out << std::endl;
    }
    out.close();
}

void destroy()
{
    delete[] table;
}

void testvocab()
{
    train_corpus_file = "corpus/minimal.txt";
    save_vocab_file = "vocab/minimal_vocab.txt";
    loadFromVocabFile(train_corpus_file);
    saveVocab();
}

void testtable()
{
    initUnigramDistribuiton();
    std::map<int, int> count_map;
    for (size_t i = 0; i < table_size; i++)
    {
        std::cout << "Table[" << i << "] = " << table[i] << std::endl;
        count_map[table[i]]++;
    }
    for (const auto &pair : count_map)
    {
        std::cout << "Word index: " << pair.first << " Count in table: " << pair.second << std::endl;
    }
}

int main()
{
    std::cout << "testing voacab" << std::endl;
    testvocab();
    std::cout << "testing table" << std::endl;
    testtable();

    destroy();
    return 0;
}