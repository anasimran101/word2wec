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


std::unordered_map<std::string, int> vocab_hash;
std::vector<word> vocab_list;

std::string train_corpus_file, output_file;
std::string save_vocab_file, read_vocab_file;
struct word *vocab;
std::ifstream fin;

int window = 5, min_count = 5, num_threads = 4, min_reduce = 1;

int vocab_size = 0, layer1_size = 100, layer1_size_aligned = ((layer1_size + 15) / 16) * 16;
long long train_words = 0, word_count_actual = 0, file_size = 0;
int epochs = 5;
float alpha = 0.025, starting_alpha, sample = 1e-3;
float *syn0;
int *sen;
clock_t start;
size_t table_size = 100;
int *table;
bool binary = false;
int negative = 5;
void testsen();
// calculate wj^0.75 / SUM i= 0->n (w_i^0.75)
void initUnigramDistribuiton()
{
    table = new int[table_size];
    float a = 0.75;
    float sum_p = 0.0, p_w = 0.0;
    size_t index = 0, prev_index = 0;
    for (size_t i = 0; i < vocab_list.size(); i++)
        sum_p += pow(vocab_list[i].count, a);
    for (size_t i = 0; i < vocab_list.size(); i++)
    {
        p_w += pow(vocab_list[i].count, a) / sum_p;
        prev_index = index;
        index = p_w * table_size;
        //std::cout << "Word: " << vocab_list[i].word << " Prob: " << p_w << " Index: " << index << std::endl;
        for (size_t j = prev_index; j < index && j < table_size; j++)
        {
            table[j] = i;
        }
    }
    while (index < table_size)
        table[index++] = vocab_list.size() - 1;
}

void InitNet() {
    int a, b;
    unsigned int next_random = 1;
    a = posix_memalign((void **)&syn0, 128, (int)vocab_size * layer1_size_aligned * sizeof(float));
    if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}

    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned int)1664525 + 1013904223;
        syn0[a * layer1_size_aligned + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
    }
}


void logPrameters() {
    std::cout << "Training model" << std::endl;
    std::cout << "Vocab size: " << vocab_size << std::endl;
    std::cout << "Train File Size: " << file_size << " bytes" << std::endl;
    std::cout << "Vector size: " << layer1_size << std::endl;
    std::cout << "Alpha: " << alpha << std::endl;
    std::cout << "Words in train file: " << train_words << std::endl;
    std::cout << "Table size: " << table_size << std::endl;
    std::cout << "Layer1 size: " << layer1_size << std::endl;
    std::cout << "Window size: " << window << std::endl;
    std::cout << "Negative samples: " << negative << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
}

void *TrainModelThread(void *id)
{
    int word_index, sentence_length = 0;
    long long word_count = 0, last_word_count = 0;
    int local_iter = epochs; //check
    unsigned int next_random = (long)id;
    int sentence_num;
    clock_t now;
    //aplha array is at the end of sen array
    float *alpha_ptr = (float *)sen + MAX_SENTENCE_NUM * MAX_SENTENCE_LENGTH;
    std::ifstream fi(train_corpus_file, std::ios::in);
    fi.seekg(std::ios::pos_type(file_size / (int)num_threads * (long)id));
    sentence_length = 0;
    sentence_num = 0;
    std::string word;
    while (1)
    {
        if (word_count - last_word_count > 5)
        {
            // No mutual exclusion here for word_count_actual, which is ok (hogwild)
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            now = clock();
            std::cout << "Alpha: " << alpha << "  Progress: " << ((word_count_actual / (float)(epochs * train_words + 1)) * 100) 
            << "%  Words/thread/sec: " << (word_count_actual / ((float)(now - start + 1) / (float)CLOCKS_PER_SEC * 1000)) << "k\n";
            std::cout.flush();
            //decay alpha as sarting_aplha * progress_ratio
            alpha = starting_alpha * (1 - word_count_actual / (float)(epochs * train_words + 1));
            if (alpha < starting_alpha * 0.0001)
                alpha = starting_alpha * 0.0001;
        }
        while (1)
        {
            if(loadWordFromFile(word, fi) == 0)
                break;
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
                //sample frequent words
                float v = (float)vocab_list[word_index].count / (sample * train_words);
                float ran = (sqrt(v) + 1.0f) / v;
                //psuedorandom number generator xn+1 = (a * xn + c) mod m
                next_random = next_random * (unsigned int)1664525 + 1013904223;
                // convert to [0,1] using lower 16 bits.
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
        testsen();
            

        // Do GPU training here
        trainGpu(sentence_num);
        //////////////////////
        sentence_num = 0;
        sentence_length = 0;

        if (loadWordFromFile(word, fi) == 0 || (word_count > train_words / num_threads))
        {
            std::cout << "Thread " << (long)id << " completed an epoch" << std::endl;
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
    long a, b;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    std::cout << "Starting training using file " << train_corpus_file << std::endl;
    starting_alpha = alpha;
    if (!read_vocab_file.empty())

        loadFromVocabFile(read_vocab_file);
    else
        loadFromTrainFile(train_corpus_file);
    if (!save_vocab_file.empty())
        saveVocab();
    if (output_file.empty())
        return;
    
    InitNet();
    initUnigramDistribuiton();
    logPrameters();
    initGpu();

    
    start = clock();
    for (a = 0; a < num_threads; a++)
        pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++)
        pthread_join(pt[a], NULL);
    freeGpu();
    // Save  the word_index vectors

    std::ofstream out(output_file, binary ? std::ios::binary : std::ios::out);
    

    out << vocab_size << " " << layer1_size << std::endl;
    for (size_t i = 0; i < vocab_size; i++)
    {
        out << vocab_list[i].word << " ";
        if (binary)
            for (b = 0; b < layer1_size; b++)
                out << syn0[i * layer1_size_aligned + b] << " ";
        else
            for (b = 0; b < layer1_size; b++)
                out << syn0[i * layer1_size_aligned + b] << " ";
        out << std::endl;
    }
    out.close();
    free(pt);
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

void testsen(){
    for (int i = 0; i < MAX_SENTENCE_NUM; i++)
    {
        if(sen[i * MAX_SENTENCE_LENGTH] == 0)
            continue;
        std::cout << "Sentence " << i << ": ";
        for (int j = 0; j < MAX_SENTENCE_LENGTH; j++)
            if (sen[i * MAX_SENTENCE_LENGTH + j] != -1)
            {
                int k =sen[i * MAX_SENTENCE_LENGTH + j];
                if (k < vocab_list.size())
                    std::cout << vocab_list[k].word << " ";
            }
            std::cout << std::endl;
    }
    std::cout << std::endl;
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
    std::cout << __FILE__ << " "  << __LINE__ << std::endl;
    std::cout << "testing voacab" << std::endl;
    //testvocab();
    std::cout << "testing table" << std::endl;
    //testtable();

    std::cout << "Training model" << std::endl;
    train_corpus_file = "corpus/minimal.txt";
    output_file = "vectors.txt";
    read_vocab_file="";
    save_vocab_file = "vocab/minimal_vocab.txt";
    TrainModel();
    destroy();
    return 0;
}