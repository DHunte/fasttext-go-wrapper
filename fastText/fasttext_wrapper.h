#ifndef FASTTEXT_WRAPPER
#define FASTTEXT_WRAPPER
int ft_load_model(const char *path);
int ft_predict(const char *query_in, float *prob, char *out, int out_size);
int ft_get_vector_dimension();
int ft_get_sentence_vector(const char *query_in, float *vector, int vector_size);
#endif
