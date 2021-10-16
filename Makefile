CC=nvcc
EXEC=app
SRC= $(wildcard src/*.cu)
OBJ= $(SRC:.c=.o)
LDFLAGS=-lsfml-graphics -lsfml-window -lsfml-system

all: $(EXEC)

$(EXEC): $(OBJ)
	@$(CC) -o bin/$@ $^ $(LDFLAGS)

%.o: %.c
	@$(CC) -o obj/$@ -c $< $(CFLAGS)

