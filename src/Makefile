######################################################################
#                            Makefile
#
# by: Toful
######################################################################

# General defines
CC = cc

SRC = backPropagation.c

BIN = backPropagation.obj

CFLAGS = -lm

######################################################################
all :
		$(CC) -Wall -O3 $(SRC) $(CFLAGS) -o $(BIN)

clean :
		rm -rf $(BIN)


run : clean all
		./$(BIN) "data.txt" 10000 89 0.05 0.25 2 5 1

test : clean all
		./$(BIN) "data_test.txt" 1000 80 0.05 0.25 2 5 1

#./backPropagation.obj "data.txt" 650 89 0.004 0.005 2 5 1
#./backPropagation.obj "data.txt" 270 89 0.02 0.01 2 5 1
#./backPropagation.obj "data_test.txt" 25000 89 0.002 0.001 2 5 1
#./backPropagation.obj "data_test.txt" 1000 89 0.02 0.005 2 5 1
#./backPropagation.obj "data.txt" 20000 89 0.02 0.1 6 4 8 10 10 8 1