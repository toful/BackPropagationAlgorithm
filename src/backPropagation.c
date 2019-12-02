#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

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

    double *** accum_update_w; //update weights matrix for the batch bp
    double ** accum_update_theta; //update theta values for the batch bp
} neuralNetwork_struct;

dataset_struct train_dataset; //train dataset storing structure
dataset_struct test_dataset; //test dataset storing structure
neuralNetwork_struct nn; //neural network structure

char * arguments = "The following input values are needed:\n\tTraining Dataset file\n\tTest Dataset file\n\tNumber of epochs\n\0t\% of the dataset used as the training set\n\tLearning rate(n)\n\tMomentum (alpha)\n\tNumber of hidden layers (including the output layer)\n\t[neurons for each layer]";



void print_dataset( dataset_struct dataset ){
    for(int m=0; m<dataset.num_samples; m++){
        for(int n=0; n<dataset.num_features; n++)
            printf("%lf\t", dataset.x_values[m][n]);
        for(int n=0; n<dataset.num_outputs; n++)
            printf("%lf\t", dataset.y_values[m][n]);
        printf("\n");
    }
}

void load_file( FILE * myFile, dataset_struct * dataset, int num_samples, int num_features, int num_outputs ){
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
            dataset->x_values[m][n] = tmp[m][n];
        }
        for( k=0; k< num_outputs; k++){
            dataset->y_values[m][k] = tmp[m][num_features+k];
        }
        free( tmp[m] );
    }
    free( tmp );
    printf("Dataset loaded\n");
}

void init_dataset( char * input_file, dataset_struct * dataset ){
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
    dataset->num_samples = num_samples;
    dataset->num_features = num_features;
    dataset->num_outputs = num_outputs;

    dataset->x_values = ( double ** ) malloc( sizeof( double * ) * num_samples );
    for( int i = 0; i < num_samples; i++ )
        dataset->x_values[i] = ( double * ) malloc( sizeof( double ) * num_features );

    dataset->y_values = ( double ** ) malloc( sizeof( double * ) * num_samples );
    for( int i = 0; i < num_samples; i++ )
        dataset->y_values[i] = ( double *) malloc( sizeof( double ) * num_outputs );

    dataset->x_min_values = ( double * ) malloc( sizeof( double ) * num_features );
    dataset->x_max_values = ( double * ) malloc( sizeof( double ) * num_features );
    dataset->y_min_values = ( double * ) malloc( sizeof( double ) * num_outputs );
    dataset->y_max_values = ( double * ) malloc( sizeof( double ) * num_outputs );
    printf("Start Reading data from %s\n", input_file );
    load_file( myFile, dataset, num_samples, num_features, num_outputs);
}

void init_nn( dataset_struct dataset ){
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

    //INITIALIZING THE ACCUM UPDATES OF THE WEIGHTS
    nn.accum_update_w = (double ***) malloc( sizeof(double **) * nn.hidden_layers ); //num hidden layers +1 because of the output layer
    //initializing the weigths between the input layer and the first hidden layer
    nn.accum_update_w[0] = (double **) malloc( sizeof(double *) * nn.neurons_for_hidden_layer[0] );
    for( i=0; i < nn.neurons_for_hidden_layer[0]; i++ ){
        nn.accum_update_w[0][i] = (double *) calloc( dataset.num_features, sizeof(double) );
    }
    //initializing the weights between hidden layers
    for( l=1; l < nn.hidden_layers; l++ ){
        nn.accum_update_w[l] = (double **) malloc( sizeof(double *) * nn.neurons_for_hidden_layer[l] );
        for( i=0; i < nn.neurons_for_hidden_layer[l]; i++ ){
            nn.accum_update_w[l][i] = (double *) calloc( nn.neurons_for_hidden_layer[l-1], sizeof(double) );
        }
    }

    //INITIALIZING THE ACCUM UPDATE THETA VALUES
    nn.accum_update_theta = (double **) malloc( sizeof(double *) * nn.hidden_layers );
    for( l=0; l < nn.hidden_layers; l++ ){
        nn.accum_update_theta[l] = (double *) calloc( nn.neurons_for_hidden_layer[l], sizeof(double) );
    }
}

void reset_nn( dataset_struct dataset ){
    int l, i, j;
    //INITIALIZING THE WEIGHTS
    //initializing the weigths between the input layer and the first hidden layer
    l=0;
    for( i=0; i < nn.neurons_for_hidden_layer[l]; i++ ){
        for( j=0; j<dataset.num_features; j++ )
            nn.w[l][i][j] = (double)rand()/RAND_MAX;
    }
    //initializing the weights between hidden layers
    for( l=1; l < nn.hidden_layers; l++ ){
        for( i=0; i < nn.neurons_for_hidden_layer[l]; i++ ){
            for( j=0; j < nn.neurons_for_hidden_layer[ l-1 ]; j++ )
                nn.w[l][i][j] = (double)rand()/RAND_MAX;
        }
    }

    //INITIALIZING THE THETA VALUES
    for( l=0; l < nn.hidden_layers; l++ ){
        for( i=0; i<nn.neurons_for_hidden_layer[l]; i++ )
            nn.theta[l][i] = (double)rand()/RAND_MAX;
    }

    //INITIALIZING THE UPDATES OF THE WEIGHTS
    //initializing the weigths between the input layer and the first hidden layer
    l=0;
    for( i=0; i < nn.neurons_for_hidden_layer[0]; i++ ){
        for( j=0; j<dataset.num_features; j++ )
            nn.update_w[0][i][j] = 0;
    }
    //initializing the weights between hidden layers
    for( l=1; l < nn.hidden_layers; l++ ){
        for( i=0; i < nn.neurons_for_hidden_layer[l]; i++ ){
            for( j=0; j < nn.neurons_for_hidden_layer[ l-1 ]; j++ )
                nn.update_w[l][i][j] = 0;
        }
    }

    //INITIALIZING THE UPDATE THETA VALUES
    for( l=0; l < nn.hidden_layers; l++ ){
        for( i=0; i<nn.neurons_for_hidden_layer[l]; i++ )
            nn.update_theta[l][i] = 0;
    }
}

double * feed_forward_propagation( ){
    double aux;
    int i, j, l;
    //computing the activation values of the first hidden layer
    l=0;
    for( i=0; i < nn.neurons_for_hidden_layer[l] ;i++ ){
        aux = 0;
        for( j=0; j < train_dataset.num_features; j++)
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
            nn.d[l][j] = compute_derivate_a( nn.h[l+1][j] )*aux; //ERROR found (using h0 which doesn't exists)
        }
    }
}

void update_nn( double n, double alpha){
    int l, i, j; 
    //updating the weights and thresholds of the first hidden layer
    l=0;
    for( i=0; i < nn.neurons_for_hidden_layer[l] ;i++ ){
        for( j=0; j < train_dataset.num_features; j++){
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

void online_BP_algorithm( int epochs, int trainset_size, double n, double alpha, double s_min, double s_max){
    int i, j, m, epoch;
    double * y_pred;
    double err, class_err;
    int tp, tn, fp, fn;
    //For epoch = 1 To num epochs
    for( epoch = 0; epoch < epochs; epoch++ ){
        err =0;
        tp=0; tn=0; fp=0; fn=0;
        //For pat = 1 To num training patterns
        for( i = 0; i < trainset_size; i++ ){
            //Choose a random pattern of the training set
            m = rand()%trainset_size;
            nn.a[0] = train_dataset.x_values[m];
            //Feed−forward propagation of pattern x μ to obtain the output o(x μ )
            feed_forward_propagation();
            //Back−propagate the error for this pattern
            error_back_propagation( train_dataset.y_values[m] );
            //Update the weights and thresholds
            update_nn( n, alpha ); //n and alpha
        }
        for( i = 0; i < train_dataset.num_samples; i++ ){
            nn.a[0] = train_dataset.x_values[i];
            y_pred = feed_forward_propagation();
            for( j = 0; j < train_dataset.num_outputs; j++){
                if( y_pred[j] < 0.5 ){
                    if( train_dataset.y_values[i][j] == 0 ) tn++;
                    else fn++;
                }
                else{
                    if( train_dataset.y_values[i][j] == 1 ) tp++;
                    else fp++;
                }
                err += pow( (y_pred[j] - train_dataset.y_values[i][j]), 2);
            }
        }
        err = err/2;
        class_err = (float)(fp + fn) / (float)(tp + tn + fp + fn );
        //printf("\nConfusion Matrix\n%d\t%d\n%d\t%d\n", tp, fp, fn, tn);
        printf("Epoch: %d \tQuadratic error: %lf\tClassification error: %lf\n", epoch, err, class_err );
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double test_BP_algorithm( int trainset_size, double s_min, double s_max, char * output_file){
    FILE * outFile;
    double * y_pred, y, class_err;
    int i, j;
    int tp=0, tn=0, fp=0, fn=0;

    //opening the output file
    outFile = fopen( output_file, "wb" );
    //Feed−forward all test patterns
    for( i = 0; i < test_dataset.num_samples; i++ ){
        nn.a[0] = test_dataset.x_values[i];
        y_pred = feed_forward_propagation();
        y=0;
        for( j = 0; j < train_dataset.num_outputs; j++){
            if( y_pred[j] < 0.5 ){
                y = 0;
                if( test_dataset.y_values[i][j] == 0 ) tn++;
                else fn++;
            }
            else{
                y = 1;
                if( test_dataset.y_values[i][j] == 1 ) tp++;
                else fp++;
            }
        }
        //printf("%lf, %lf\n", test_dataset.y_values[i][0], y );
        fprintf(outFile, "%lf, %lf, %lf, %lf\n", test_dataset.y_values[i][0], y, test_dataset.x_values[i][0], test_dataset.x_values[i][1] );
    }
    class_err = (float)(fp + fn) / (float)(tp + tn + fp + fn );
    printf("\nConfusion Matrix\n%d\t%d\n%d\t%d\n", tp, fp, fn, tn);
    printf("\nClassification error over the TestSet: %lf\n", class_err );
    fclose( outFile );
    return( class_err );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    srand ( time ( NULL));
    if( argc < 5){
        printf("Too few arguments.\n");
        printf("%s\n", arguments);
        exit(1);
    }
    double s_min = 0.1;
    double s_max = 0.9;
    int epochs = atoi(argv[3]); //number of epochs
    float trainset_size_perc = atof(argv[4]); //% of the dataset used as the training set 
    double n = atof(argv[5]); //learning rate: between 0.2 and 0.01
    double alpha = atof(argv[6]); //momentum: between 0.9 and 0.0

    //genrating the dataset
    init_dataset( argv[1], & train_dataset );
    //print_dataset( train_dataset );
    init_dataset( argv[2], &test_dataset );
    //print_dataset( test_dataset );
    
    //initializing the neural network
    nn.hidden_layers = atoi(argv[7]);
    if( argc < 7 + nn.hidden_layers ){
        printf("Too few arguments.\n");
        printf("%s\n", arguments);
        exit(1);
    }
    nn.neurons_for_hidden_layer = (int *) malloc( sizeof(int) * nn.hidden_layers );
    for( int i = 0; i < nn.hidden_layers; i++ )
        nn.neurons_for_hidden_layer[i] = atoi( argv[7+i] );
    init_nn( train_dataset );

    struct stat st = {0};
    if (stat("out/", &st) == -1) {
        mkdir("out/", 0700);
    }

    char on_test[200] = "out/online_";
    strcat( on_test, &argv[1][10]);
    strcat( on_test, ".csv");


    //ONLINE BP ALGORITHM
    //////////////////////////////////////////////////////////////////

    //TRAINING PHASE
    int trainset_size = FLOAT_TO_INT( train_dataset.num_samples * trainset_size_perc / 100 );
    //reset_nn();
    //TRAINING PHASE
    online_BP_algorithm( epochs, trainset_size, n, alpha, s_min, s_max);
    //TESTING PHASE
    test_BP_algorithm( trainset_size, s_min, s_max, on_test);

 
    exit( 0 );
}