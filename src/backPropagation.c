#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define FLOAT_TO_INT(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))

typedef struct {
    double ** values;
    double ** x_values;
    double ** y_values;
    int num_samples;
    int num_features;
    int num_outputs;
    double * x_min_values;
    double * x_max_values;
    double * y_min_values;
    double * y_max_values;
} dataset_struct;

typedef struct {
    int hidden_layers;
    int * neurons_for_hidden_layer;
    double *** w; //weights matrix
    double ** theta; //theta values
    double ** h;
    double ** a; //activation values = g(h)
    double ** d; //gradien error

    double *** update_w; //update weights matrix
    double ** update_theta; //update theta values
} neuralNetwork_struct;

dataset_struct dataset; //dataset storing structure
neuralNetwork_struct nn; //neural network structure

char * arguments = "The following input values are needed:\n\tData file\n\tLearining rate(n)\n\tMomentum (alpha)\n\tNumber of hidden layers (including the output layer)\n\t[neurons for each layer]";


void print_dataset(){
    for(int m=0; m<dataset.num_samples; m++){
        for(int n=0; n<dataset.num_features; n++)
            printf("%lf\t", dataset.x_values[m][n]);
        for(int n=0; n<dataset.num_outputs; n++)
            printf("%lf\t", dataset.y_values[m][n]);
        printf("\n");
    }
}

void load_file( FILE * myFile, int num_samples, int num_features, int num_outputs ){
    int m, n, k;
    double ** tmp = ( double ** ) malloc( sizeof( double * ) * num_samples );
    for( m = 0; m < num_samples; m++ ){
        tmp[m] = ( double * ) malloc( sizeof( double ) * (num_features + num_outputs) );
    }

    m = 0;
    n = 0;
    while( fscanf(myFile, "%lf", &tmp[m][n++] ) != EOF ){
        if( n >= num_features + num_outputs ){
            n = 0;
            m++;
        }
    }
    fclose(myFile);

    for( m=0; m < num_samples; m++){
        for( n=0; n<num_features; n++){
            dataset.x_values[m][n] = tmp[m][n];
        }
        for( k=0; k< num_outputs; k++){
            dataset.y_values[m][k] = tmp[m][num_features+k];
        }
        free( tmp[m] );
    }
    free( tmp );
    printf("Dataset loaded\n");
}

void init_dataset( char * input_file ){
    FILE *myFile;
    int num_samples, num_features, num_outputs;
    //opening the input file
    myFile = fopen( input_file, "r" );
    if (myFile == NULL) {
        printf("ERROR: failed to open file %s\n", input_file );
        exit(1);
    }
    //reading the number of samples and feature for sample in the file
    if( fscanf(myFile, "%d", &num_samples ) == EOF ||
        fscanf(myFile, "%d", &num_features ) == EOF ||
        fscanf(myFile, "%d", &num_outputs ) == EOF){
        printf("ERROR: file %s is not in the proper format\n", input_file );
        exit(1);
    }
    dataset.num_samples = num_samples;
    dataset.num_features = num_features;
    dataset.num_outputs = num_outputs;

    dataset.x_values = ( double ** ) malloc( sizeof( double * ) * num_samples );
    for( int i = 0; i < num_samples; i++ )
        dataset.x_values[i] = ( double * ) malloc( sizeof( double ) * num_features );

    dataset.y_values = ( double ** ) malloc( sizeof( double * ) * num_samples );
    for( int i = 0; i < num_samples; i++ )
        dataset.y_values[i] = ( double *) malloc( sizeof( double ) * num_outputs );

    dataset.x_min_values = ( double * ) malloc( sizeof( double ) * num_features );
    dataset.x_max_values = ( double * ) malloc( sizeof( double ) * num_features );
    dataset.y_min_values = ( double * ) malloc( sizeof( double ) * num_outputs );
    dataset.y_max_values = ( double * ) malloc( sizeof( double ) * num_outputs );
    printf("Start Reading data from %s\n", input_file );
    load_file( myFile, num_samples, num_features, num_outputs);
    //print_dataset();
}

void scale_dataset( double s_min, double s_max ){
    printf("Scaling the dataset to range [%f, %f]\n", s_min, s_max);
    double max, min;
    for( int n=0; n < dataset.num_features; n++){
        max = LONG_MIN;
        min = LONG_MAX;
        for( int m=0; m < dataset.num_samples; m++){
            if( dataset.x_values[m][n] > max ) max = dataset.x_values[m][n];
            if( dataset.x_values[m][n] < min ) min = dataset.x_values[m][n];
        }
        dataset.x_min_values[n] = min;
        dataset.x_max_values[n] = max;
        for( int m=0; m < dataset.num_samples; m++)
            dataset.x_values[m][n] = s_min + (s_max - s_min)/(max - min)*(dataset.x_values[m][n] - min);
    }
    for( int n=0; n < dataset.num_outputs; n++){
        max = LONG_MIN;
        min = LONG_MAX;
        for( int m=0; m < dataset.num_samples; m++){
            if( dataset.y_values[m][n] > max ) max = dataset.y_values[m][n];
            if( dataset.y_values[m][n] < min ) min = dataset.y_values[m][n];
        }
        dataset.y_min_values[n] = min;
        dataset.y_max_values[n] = max;
        for( int m=0; m < dataset.num_samples; m++)
            dataset.y_values[m][n] = s_min + (s_max - s_min)/(max - min)*(dataset.y_values[m][n] - min);
    }
    //print_dataset();
}

void descale_dataset( double s_min, double s_max ){
    printf("Descaling the dataset from range [%f, %f]\n", s_min, s_max);
    double max, min;
    for( int n=0; n < dataset.num_features; n++){
        min = dataset.x_min_values[n];
        max = dataset.x_max_values[n];
        for( int m=0; m < dataset.num_samples; m++)
            dataset.x_values[m][n] = min + (max - min)/(s_max - s_min)*(dataset.x_values[m][n] - s_min);
    }
    for( int n=0; n < dataset.num_outputs; n++){
        min = dataset.y_min_values[n];
        max = dataset.y_max_values[n];
        for( int m=0; m < dataset.num_samples; m++)
            dataset.y_values[m][n] = min + (max - min)/(s_max - s_min)*(dataset.y_values[m][n] - s_min);
    }
    //print_dataset();
}

double descale_x_value( double value, int feature, double s_min, double s_max ){
    double x_max, x_min;
    x_min = dataset.x_min_values[ feature ];
    x_max = dataset.x_max_values[ feature ];
    return ( x_min + (x_max - x_min)/(s_max - s_min)*(value - s_min) );
}

double descale_y_value( double value, int feature, double s_min, double s_max ){
    double y_max, y_min;
    y_min = dataset.y_min_values[feature];
    y_max = dataset.y_max_values[feature];
    return ( y_min + (y_max - y_min)/(s_max - s_min)*(value - s_min) );
}

void init_nn(){
    int l, i, j;
    //INITIALIZING THE WEIGHTS
    nn.w = (double ***) malloc( sizeof(double **) * (nn.hidden_layers + 1) ); //num hidden layers +1 because of the output layer
    //initializing the weigths between the input layer and the first hidden layer
    l=0;
    nn.w[l] = (double **) malloc( sizeof(double *) * nn.neurons_for_hidden_layer[l] );
    for( i=0; i < nn.neurons_for_hidden_layer[l]; i++ ){
        nn.w[l][i] = (double *) calloc( dataset.num_features, sizeof(double) );
        for( j=0; j<dataset.num_features; j++ )
            nn.w[l][i][j] = (double)rand()/RAND_MAX;
    }
    //initializing the weights between hidden layers
    for( l=1; l < nn.hidden_layers; l++ ){
        nn.w[l] = (double **) malloc( sizeof(double *) * nn.neurons_for_hidden_layer[l] );
        for( i=0; i < nn.neurons_for_hidden_layer[l]; i++ ){
            nn.w[l][i] = (double *) calloc( nn.neurons_for_hidden_layer[ l-1 ], sizeof(double) );
            for( j=0; j < nn.neurons_for_hidden_layer[ l-1 ]; j++ )
                nn.w[l][i][j] = (double)rand()/RAND_MAX;
        }
    }

    //INITIALIZING THE THETA VALUES
    nn.theta = (double **) malloc( sizeof(double *) * (nn.hidden_layers) );
    for( l=0; l < nn.hidden_layers; l++ ){
        nn.theta[l] = (double *) malloc( sizeof(double) * nn.neurons_for_hidden_layer[l] );
        for( i=0; i<nn.neurons_for_hidden_layer[l]; i++ )
            nn.theta[l][i] = (double)rand()/RAND_MAX;
    }

    //INITIALIZING h MATRICES
    nn.h = (double **) malloc( sizeof(double *) * (nn.hidden_layers + 1) );
    nn.h[0] = (double *) malloc( sizeof(double) * dataset.num_features );
    for( l=1; l < nn.hidden_layers + 1; l++ )
        nn.h[l] = (double *) malloc( sizeof(double) * nn.neurons_for_hidden_layer[l-1] );

    //INITIALIZING THE ACTIVATION FUNCTION RESULT MATRICES g
    nn.a = (double **) malloc( sizeof(double *) * (nn.hidden_layers + 1) );
    nn.a[0] = (double *) malloc( sizeof(double) * dataset.num_features );
    for( l=1; l < nn.hidden_layers + 1; l++ )
        nn.a[l] = (double *) malloc( sizeof(double) * nn.neurons_for_hidden_layer[l-1] );

    //INITIALIZING THE GRADIEN ERROR MATRICES
    nn.d = (double **) malloc( sizeof(double *) * nn.hidden_layers );
    for( l=0; l < nn.hidden_layers; l++ )
        nn.d[l] = (double *) malloc( sizeof(double) * nn.neurons_for_hidden_layer[l] );

    //INITIALIZING THE UPDATES OF THE WEIGHTS
    nn.update_w = (double ***) malloc( sizeof(double **) * nn.hidden_layers ); //num hidden layers +1 because of the output layer
    //initializing the weigths between the input layer and the first hidden layer
    nn.update_w[0] = (double **) malloc( sizeof(double *) * nn.neurons_for_hidden_layer[0] );
    for( i=0; i < nn.neurons_for_hidden_layer[0]; i++ ){
        nn.update_w[0][i] = (double *) calloc( dataset.num_features, sizeof(double) );
    }
    //initializing the weights between hidden layers
    for( l=1; l < nn.hidden_layers; l++ ){
        nn.update_w[l] = (double **) malloc( sizeof(double *) * nn.neurons_for_hidden_layer[l] );
        for( i=0; i < nn.neurons_for_hidden_layer[l]; i++ ){
            nn.update_w[l][i] = (double *) calloc( nn.neurons_for_hidden_layer[l-1], sizeof(double) );
        }
    }

    //INITIALIZING THE UPDATE THETA VALUES
    nn.update_theta = (double **) malloc( sizeof(double *) * nn.hidden_layers );
    for( l=0; l < nn.hidden_layers; l++ ){
        nn.update_theta[l] = (double *) calloc( nn.neurons_for_hidden_layer[l], sizeof(double) );
    }
}

double * feed_forward_propagation(){
    double aux;
    int i, j, l;
    //computing the activation values of the first hidden layer
    l=0;
    for( i=0; i < nn.neurons_for_hidden_layer[l] ;i++ ){
        aux = 0;
        for( j=0; j < dataset.num_features; j++)
            aux += nn.w[l][i][j] * nn.a[l][j];
        nn.h[l+1][i] = aux - nn.theta[l][i];  //=h(l)i
        nn.a[l+1][i] = 1 / ( 1 + exp(- nn.h[l+1][i] ) ); //g(h)
    }

    //computing the activation values of intermediate hidden layers
    for( l=1;  l < nn.hidden_layers; l++){
        for( i=0; i < nn.neurons_for_hidden_layer[l]; i++ ){
            aux = 0;
            for( j=0; j < nn.neurons_for_hidden_layer[l-1]; j++){
                aux += nn.w[l][i][j] * nn.a[l][j];
            }
            nn.h[l+1][i] = aux - nn.theta[l][i];  //=h(l)i
            nn.a[l+1][i] = 1 / ( 1 + exp(- nn.h[l+1][i] ) ); //g(h)
        }
    }
    return nn.a[ nn.hidden_layers ]; 
}

double compute_derivate_a( double h){
    double g;
    g = 1 / ( 1 + exp(-h) );
    return (g*(1-g)); 
}

void error_back_propagation( double * y ){
    int i, j, l;
    double aux;
    //computing the error of the output layer
    l = nn.hidden_layers-1;
    aux = 0;
    for(i=0; i< nn.neurons_for_hidden_layer[l]; i++ ){
        nn.d[l][i] = compute_derivate_a( nn.h[l+1][i] )*( nn.a[l+1][i] - y[i] );
    }

    //computing the error for the rest of the layers
    for( l=nn.hidden_layers - 2; l>=0; l--){
        for(j=0; j < nn.neurons_for_hidden_layer[l]; j++){
            aux = 0;
            for(i=0; i< nn.neurons_for_hidden_layer[l+1]; i++ ){
                aux += nn.d[l+1][i] * nn.w[l+1][i][j];
            }
            nn.d[l][j] = compute_derivate_a( nn.h[l][j] )*aux;
        }
    }
}

void update_nn(double n, double alpha){
    int l, i, j; 
    //updating the weights and thresholds of the first hidden layer
    l=0;
    for( i=0; i < nn.neurons_for_hidden_layer[l] ;i++ ){
        for( j=0; j < dataset.num_features; j++){
            nn.update_w[l][i][j] = (-n * nn.d[l][i] * nn.a[l][j] ) + ( alpha * nn.update_w[l][i][j] );
            nn.w[l][i][j] = nn.w[l][i][j] + nn.update_w[l][i][j];
        }
        nn.update_theta[l][i] = (n * nn.d[l][i] ) + ( alpha * nn.update_theta[l][i] );
        nn.theta[l][i] = nn.theta[l][i] + nn.update_theta[l][i];
    }

    //updating the weights and thresholds of intermediate hidden layers
    for( l=1;  l < nn.hidden_layers; l++){
        for( i=0; i < nn.neurons_for_hidden_layer[l]; i++ ){
            for( j=0; j < nn.neurons_for_hidden_layer[l-1]; j++){
                nn.update_w[l][i][j] = (-n * nn.d[l][i] * nn.a[l][j] ) + ( alpha * nn.update_w[l][i][j] );
                nn.w[l][i][j] = nn.w[l][i][j] + nn.update_w[l][i][j];
            }
            nn.update_theta[l][i] = (n * nn.d[l][i] ) + ( alpha * nn.update_theta[l][i] );
            nn.theta[l][i] = nn.theta[l][i] + nn.update_theta[l][i];
        }
    }
}

int main(int argc, char *argv[])
{
    srand ( time ( NULL));
    if( argc < 4){
        printf("Too few arguments.\n");
        printf("%s\n", arguments);
        exit(1);
    }
    int i, j, m, epoch;
    double s_min = 0.1;
    double s_max = 0.9;
    int epochs = atoi(argv[2]); //number of epochs
    float trainset_size_perc = atof(argv[3]); //% of the dataset used as the training set 
    double n = atof(argv[4]); //learning rate: between 0.2 and 0.01
    double alpha = atof(argv[5]); //momentum: between 0.0 and 0.9

    //genrating the dataset
    init_dataset( argv[1] );
    print_dataset();

    //preprocessing de data
    scale_dataset( s_min, s_max );
    print_dataset();
    
    //initializing the neural network
    nn.hidden_layers = atoi(argv[6]);
    if( argc < 6 + nn.hidden_layers ){
        printf("Too few arguments.\n");
        printf("%s\n", arguments);
        exit(1);
    }
    nn.neurons_for_hidden_layer = (int *) malloc( sizeof(int) * nn.hidden_layers );
    for( i = 0; i < nn.hidden_layers; i++ )
        nn.neurons_for_hidden_layer[i] = atoi( argv[6+i] );
    init_nn();

    //TRAINING PHASE
    int trainset_size = FLOAT_TO_INT( dataset.num_samples * trainset_size_perc / 100 );
    double * y_pred;
    double err, aux1, aux2;
    err =0; aux1=0; aux2=0;
    //For epoch = 1 To num epochs
    for( epoch = 0; epoch < epochs; epoch++ ){
        //For pat = 1 To num training patterns
        for( i = 0; i < trainset_size; i++ ){
            //Choose a random pattern of the training set
            m = rand()%trainset_size;
            nn.a[0] = dataset.x_values[m];
            //Feed−forward propagation of pattern x μ to obtain the output o(x μ )
            feed_forward_propagation();
            //Back−propagate the error for this pattern
            error_back_propagation( dataset.y_values[m] );
            //Update the weights and thresholds
            update_nn( n, alpha ); //n and alpha
        }
        //Feed−forward all training patterns and calculate their prediction quadratic error
        //Feed−forward all training patterns and calculate the percentage of error for the predictions
        for( i = 0; i < trainset_size; i++ ){
            nn.a[0] = dataset.x_values[i];
            y_pred = feed_forward_propagation();
            for( j = 0; j < dataset.num_outputs; j++){
                aux1 += fabs( descale_y_value( y_pred[j], j, s_min, s_max ) - descale_y_value( dataset.y_values[i][j], j, s_min, s_max ) );
                aux2 += descale_y_value( dataset.y_values[i][j], j, s_min, s_max );
                err += pow( (y_pred[j] - dataset.y_values[i][j]), 2);
            }
        }
        err = err/2;
        printf("Epoch: %d \tQuadratic error: %lf\tPercentage of error: %lf\n", epoch, err, (aux1/aux2*100) );
        //Feed−forward all validation patterns and calculate their prediction quadratic error
        //WE DON'T USE A VALIDATION SET IN THIS PRACTICE
    }

    //TESTING PHASE
    FILE * outFile;
    struct stat st = {0};
    if (stat("out/", &st) == -1) {
        mkdir("out/", 0700);
    }
    //opening the output file
    outFile = fopen( "out/result.csv", "wb" );
    double y, z; 
    //Feed−forward all test patterns
    //Descale the predictions of test patterns, and evaluate them
    for( i = trainset_size; i < dataset.num_samples; i++ ){
        nn.a[0] = dataset.x_values[i];
        y_pred = feed_forward_propagation();
        y=0; z=0;
        for( j = 0; j < dataset.num_outputs; j++){
            y += descale_y_value( y_pred[j], j, s_min, s_max );
            z += descale_y_value( dataset.y_values[i][j], j, s_min, s_max );
        }
        printf("%lf, %lf, %lf\n", z, y, fabs(z - y) );
        fprintf(outFile, "%lf, %lf, %lf\n", z, y, fabs(z - y) );
        aux1 += fabs(z - y);
        aux2 += z;
    }
    printf("\nPercentage of error over the TestSet: %lf\n", (aux1/aux2*100) );
    fclose(outFile);
    
    exit( 0 );
}