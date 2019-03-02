MAIN_DIR=main
SOURCE_DIR=sources
OBJECT_DIR=objects
EXECUTABLE_DIR=executables

SOURCES=$(shell find $(SOURCE_DIR)/ -name '*.cu')
MAINS=$(shell find $(MAIN_DIR)/ -name '*.cu')
MAIN_OBJECTS=$(MAINS:$(MAIN_DIR)/%.cu=$(OBJECT_DIR)/%.o)
OBJECTS=$(SOURCES:$(SOURCE_DIR)/%.cu=$(OBJECT_DIR)/%.o)
EXECUTABLES=$(MAINS:$(MAIN_DIR)/%.cu=%)

CC=nvcc
CFLAGS=-g -I$(SOURCE_DIR)/

all: $(EXECUTABLES)

.PHONY: clean

.PRECIOUS: $(OBJECT_DIR)/%.o

$(EXECUTABLES): $(MAIN_OBJECTS) $(OBJECTS)
	$(CC) $(OBJECT_DIR)/$@.o $(OBJECTS) -o $(EXECUTABLE_DIR)/$@

$(OBJECT_DIR)/%.o: $(MAIN_DIR)/%.cu
	$(CC) $(CFLAGS) -dc $< -o $@

$(OBJECT_DIR)/%.o: $(SOURCE_DIR)/%.cu
	$(CC) $(CFLAGS) -dc $< -o $@

clean:
# should not simply write $(OBJECT_DIR)/*
# otherwise the subfolder will cause error
	rm -f $(MAIN_OBJECTS) $(OBJECTS) $(EXECUTABLE_DIR)/*