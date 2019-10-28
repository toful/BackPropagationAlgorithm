######################################################################
#                            Makefile
#
# by: Toful
######################################################################

# General defines
CC = cc

SRC = backPropagation.c

BIN = backPropagation.obj

CFLAGS = 

######################################################################
all :
		$(CC) -Wall -O3 $(SRC) $(CFLAGS) -o $(BIN)

clean :
		rm -rf $(BIN)


run : clean all
		./$(BIN) "data_test.txt" 10 4 1 5