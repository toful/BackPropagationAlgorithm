#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

typedef struct {
    double ** values;
    int num_samples;
    int num_features;
    double * min_values;
    double * max_values;
} dataset_struct;

typedef struct {
    int hidden_layers;
    int * neurons_for_hidden_layer;
    double *** w; //weights matrix
} neuralNetwork_struct;

dataset_struct dataset; //dataset storing structure
neuralNetwork_struct nn; //neural network structure

char * arguments = "The following input values are needed:\n\tData file\n\tNumber of samples in the data file\n\tNumber of features for each sample (including the y value)\n\tNumber of hidden layers";


void print_dataset(){
    for(int m=0; m<dataset.num_samples; m++){
        for(int n=0; n<dataset.num_features; n++)
            printf("%lf\t", dataset.values[m][n]);
        printf("\n");
    }
}

void load_file( char * input_file, int num_samples, int num_features ){
    FILE *myFile;
    int m, n;
    myFile = fopen( input_file, "r" );
    if (myFile == NULL) {
        printf("ERROR: failed to open file %s\n", input_file );
        exit(1);
    }
    m = 0;
    n = 0;
    printf("Reading data from %s\n", input_file );
    while( fscanf(myFile, "%lf", &dataset.values[m][n++] ) != EOF ){
        if( n >= num_features ){
            n = 0;
            m++;
        }
    }
    fclose(myFile);
}

void init_dataset( char * input_file, int num_samples, int num_features ){
    dataset.num_samples = num_samples;
    dataset.num_features = num_features;
    dataset.values = ( double ** ) malloc( sizeof( double * ) * num_samples );
    for( int i = 0; i < num_samples; i++ ){
        dataset.values[i] = ( double * ) calloc( num_features, sizeof( double ) );
    }
    dataset.min_values = ( double * ) malloc( sizeof( double ) * num_features );
    dataset.max_values = ( double * ) malloc( sizeof( double ) * num_features );
    load_file( input_file, num_samples, num_features);
    print_dataset();
}

void scale_dataset( double s_min, double s_max ){
    printf("Scaling the dataset to range [%f, %f]\n", s_min, s_max);
    double x_max, x_min;
    for( int n=0; n < dataset.num_features; n++){
        x_max = LONG_MIN;
        x_min = LONG_MAX;
        for( int m=0; m < dataset.num_samples; m++){
            if( dataset.values[m][n] > x_max ) x_max = dataset.values[m][n];
            if( dataset.values[m][n] < x_min ) x_min = dataset.values[m][n];
        }
        dataset.min_values[n] = x_min;
        dataset.max_values[n] = x_max;
        for( int m=0; m < dataset.num_samples; m++)
            dataset.values[m][n] = s_min + (s_max - s_min)/(x_max - x_min)*(dataset.values[m][n] - x_min);
    }
    print_dataset();
}

void descale_dataset( double s_min, double s_max ){
    printf("Descaling the dataset from range [%f, %f]\n", s_min, s_max);
    double x_max, x_min;
    for( int n=0; n < dataset.num_features; n++){
        x_min = dataset.min_values[n];
        x_max = dataset.max_values[n];
        for( int m=0; m < dataset.num_samples; m++)
            dataset.values[m][n] = x_min + (x_max - x_min)/(s_max - s_min)*(dataset.values[m][n] - s_min);
    }
    print_dataset();
}

int main(int argc, char *argv[])
{
    if( argc < 4){
        printf("Too few arguments.\n");
        printf("%s\n", arguments);
        exit(1);
    }
    double s_min = 0.1;
    double s_max = 0.9;

    init_dataset( argv[1], atoi(argv[2]), atoi(argv[3]) );

    //preprocessing
    scale_dataset( s_min, s_max );

    //init neural network
    nn.hidden_layers = atoi(argv[4]);
    if( argc < 5 + nn.hidden_layers){
        printf("Too few arguments.\n");
        printf("%s\n", arguments);
        exit(1);
    }
    nn.neurons_for_hidden_layer = (int *) calloc( nn.hidden_layers, sizeof(int) );
    for( int i = 0; i < nn.hidden_layers; i++ ) nn.neurons_for_hidden_layer[i] = atoi( argv[5+i] );
    
    //feed-forward propagation


    exit( 0 );
}