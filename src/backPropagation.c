#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

typedef struct {
    double ** values;
    double ** x_values;
    double * y_values;
    int num_samples;
    int num_features;
    double * x_min_values;
    double * x_max_values;
    double y_min_values;
    double y_max_values;
} dataset_struct;

typedef struct {
    int hidden_layers;
    int * neurons_for_hidden_layer;
    double *** w; //weights matrix
    double ** theta; //theta values
    double ** a; //activation values
} neuralNetwork_struct;

dataset_struct dataset; //dataset storing structure
neuralNetwork_struct nn; //neural network structure

char * arguments = "The following input values are needed:\n\tData file\n\tNumber of samples in the data file\n\tNumber of features for each sample (without including the y value)\n\tNumber of hidden layers";


void print_dataset(){
    for(int m=0; m<dataset.num_samples; m++){
        for(int n=0; n<dataset.num_features; n++)
            printf("%lf\t", dataset.x_values[m][n]);
        printf("%lf\n", dataset.y_values[m]);
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
    double ** tmp = ( double ** ) malloc( sizeof( double * ) * num_samples );
    for( int i = 0; i < num_samples; i++ ){
        tmp[i] = ( double * ) malloc( sizeof( double ) * (num_features + 1) );
    }

    printf("Reading data from %s\n", input_file );
    while( fscanf(myFile, "%lf", &tmp[m][n++] ) != EOF ){
        if( n >= num_features + 1 ){
            n = 0;
            m++;
        }
    }
    fclose(myFile);

    for( m=0; m < num_samples; m++){
        for( n=0; n<num_features; n++){
            dataset.x_values[m][n] = tmp[m][n];
        }
        dataset.y_values[m] = tmp[m][n];
        free( tmp[m] );
    }
    free( tmp );
    printf("Dataset loaded\n");
}

void init_dataset( char * input_file, int num_samples, int num_features ){
    dataset.num_samples = num_samples;
    dataset.num_features = num_features;
    dataset.x_values = ( double ** ) malloc( sizeof( double * ) * num_samples );
    for( int i = 0; i < num_samples; i++ ){
        dataset.x_values[i] = ( double * ) malloc( sizeof( double ) * num_features );
    }
    dataset.x_min_values = ( double * ) malloc( sizeof( double ) * num_features );
    dataset.x_max_values = ( double * ) malloc( sizeof( double ) * num_features );
    dataset.y_values = ( double *) malloc( sizeof( double ) * num_samples );
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
            if( dataset.x_values[m][n] > x_max ) x_max = dataset.x_values[m][n];
            if( dataset.x_values[m][n] < x_min ) x_min = dataset.x_values[m][n];
        }
        dataset.x_min_values[n] = x_min;
        dataset.x_max_values[n] = x_max;
        for( int m=0; m < dataset.num_samples; m++)
            dataset.x_values[m][n] = s_min + (s_max - s_min)/(x_max - x_min)*(dataset.x_values[m][n] - x_min);
    }
    x_max = LONG_MIN;
    x_min = LONG_MAX;
    for( int m=0; m < dataset.num_samples; m++){
        if( dataset.y_values[m] > x_max ) x_max = dataset.y_values[m];
        if( dataset.y_values[m] < x_min ) x_min = dataset.y_values[m];
    }
    dataset.y_min_values = x_min;
    dataset.y_max_values = x_max;
    for( int m=0; m < dataset.num_samples; m++)
        dataset.y_values[m] = s_min + (s_max - s_min)/(x_max - x_min)*(dataset.y_values[m] - x_min);
    print_dataset();
}

void descale_dataset( double s_min, double s_max ){
    printf("Descaling the dataset from range [%f, %f]\n", s_min, s_max);
    double x_max, x_min;
    for( int n=0; n < dataset.num_features; n++){
        x_min = dataset.x_min_values[n];
        x_max = dataset.x_max_values[n];
        for( int m=0; m < dataset.num_samples; m++)
            dataset.x_values[m][n] = x_min + (x_max - x_min)/(s_max - s_min)*(dataset.x_values[m][n] - s_min);
    }
    x_min = dataset.y_min_values;
    x_max = dataset.y_max_values;
    for( int m=0; m < dataset.num_samples; m++)
        dataset.y_values[m] = x_min + (x_max - x_min)/(s_max - s_min)*(dataset.y_values[m] - s_min);
    print_dataset();
}

void init_nn(){
    int l, i, j;
    //INITIALIZING THE WEIGHTS
    nn.w = (double ***) malloc( sizeof(double **) * (nn.hidden_layers + 1) ); //num hidden layers +1 because of the output layer
    //initializing the weigths between the input layer and the first hidden layer
    nn.w[0] = (double **) malloc( sizeof(double *) * nn.neurons_for_hidden_layer[0] );
    for( i=0; i < nn.neurons_for_hidden_layer[0]; i++ ){
        nn.w[0][i] = (double *) calloc( dataset.num_features, sizeof(double) );
        for( j=0; j<dataset.num_features; j++ )
            nn.w[0][i][j] = (double)rand()/RAND_MAX;
    }
    //initializing the weights between hidden layers
    for( l=1; l < nn.hidden_layers; l++ ){
        nn.w[l] = (double **) malloc( sizeof(double *) * nn.neurons_for_hidden_layer[l] );
        for( i=0; i < nn.neurons_for_hidden_layer[l]; i++ ){
            nn.w[l][i] = (double *) calloc( nn.neurons_for_hidden_layer[l-1], sizeof(double) );
            for( j=0; j < nn.neurons_for_hidden_layer[l-1]; j++ )
                nn.w[l][i][j] = (double)rand()/RAND_MAX;
        }
    }
    //initializing the weigths between the last hidden layer and the output layer
    nn.w[ nn.hidden_layers ] = (double **) malloc( sizeof(double *) * 1 );
    nn.w[ nn.hidden_layers ][ 0 ] = (double *) calloc( nn.neurons_for_hidden_layer[ nn.hidden_layers - 1 ], sizeof(double) );
    for( j=0; j < nn.neurons_for_hidden_layer[ nn.hidden_layers - 1 ]; j++ )
        nn.w[ nn.hidden_layers ][ 0 ][j] = (double)rand()/RAND_MAX;

    //INITIALIZING THE THETA VALUES
    nn.theta = (double **) malloc( sizeof(double *) * (nn.hidden_layers + 1) );
    for( l=0; l < nn.hidden_layers; l++ ){
        nn.theta[l] = (double *) malloc( sizeof(double) * nn.neurons_for_hidden_layer[l] );
        for( i=0; i<nn.neurons_for_hidden_layer[l]; i++ )
            nn.theta[l][i] = (double)rand()/RAND_MAX;
    }
    nn.theta[ nn.hidden_layers ] = (double *) malloc( sizeof(double) * 1 );
    nn.theta[nn.hidden_layers][0] = (double)rand()/RAND_MAX;

    //INITIALIZING THE ACTIVATION FUNCTION RESULT MATRICES
    nn.a = (double **) malloc( sizeof(double *) * (nn.hidden_layers + 2) );
    nn.a[0] = (double *) malloc( sizeof(double) * dataset.num_features );
    for( l=1; l < nn.hidden_layers + 1; l++ )
        nn.a[l] = (double *) malloc( sizeof(double) * nn.neurons_for_hidden_layer[l-1] );
    nn.a[ nn.hidden_layers + 1 ] = (double *) malloc( sizeof(double) * 1 );
}

void feed_forward_propagation(){
    double aux;
    //computing the activation values of the first hidden layer
    int i, j, l = 0;
    for( i=0; i < nn.neurons_for_hidden_layer[0] ;i++ ){
        aux = 0;
        for( j=0; j < dataset.num_features; j++)
            aux += nn.w[l][i][j] * nn.a[l][j];
        aux = aux - nn.theta[l][i];  //=h(l)i
        aux = aux; //need to compute the activation function
        nn.a[l+1][i] = aux;
    }

    //computing the activation values of intermediate hidden layers
    for( l=1;  l < nn.hidden_layers; l++){
        for( i=0; i < nn.neurons_for_hidden_layer[l]; i++ ){
            aux = 0;
            for( j=0; j < nn.neurons_for_hidden_layer[l-1]; j++){
                aux += nn.w[l][i][j] * nn.a[l][j];
            }
            aux = aux - nn.theta[l][i];  //=h(l)i
            aux = aux; //need to compute the activation function
            nn.a[l+1][i] = aux;
        }
    }

    //computing the output
    l = nn.hidden_layers;
    aux = 0;
    for( j=0; j < nn.neurons_for_hidden_layer[ nn.hidden_layers - 1 ] ; j++)
        aux += nn.w[l][0][j] * nn.a[l][j];
    aux = aux - nn.theta[l][0];  //=h(l)i
    aux = aux; //need to compute the activation function
    nn.a[l+1][0] = aux;
}

int main(int argc, char *argv[])
{
    srand ( time ( NULL));
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
    nn.neurons_for_hidden_layer = (int *) malloc( sizeof(int) * nn.hidden_layers );
    for( int i = 0; i < nn.hidden_layers; i++ ) nn.neurons_for_hidden_layer[i] = atoi( argv[5+i] );
    init_nn();

    //feed-forward propagation
    for( int m = 0; m < dataset.num_samples*8/10; m++){
        nn.a[0] = dataset.x_values[m];
        feed_forward_propagation();

        printf("Iteration: %d Result: %lf\n", m, nn.a[ nn.hidden_layers + 1][0] );
    }

    exit( 0 );
}