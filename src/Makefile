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
		./$(BIN) "dataset/data.txt" 20000 89 0.05 0.01 3 9 5 1

test : clean all
		./$(BIN) "dataset/data_test.txt" 1000 80 0.05 0.25 2 5 1

#./$(BIN) "dataset/data.txt" 15000 89 0.05 0.25 4 9 5 5 1
#./backPropagation.obj "dataset/data.txt" 20000 89 0.02 0.1 6 4 8 10 10 8 1