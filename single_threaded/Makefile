MAIN_DIR=main
SOURCE_DIR=sources
OBJECT_DIR=objects
EXECUTABLE_DIR=executables

SOURCES=$(shell find $(SOURCE_DIR)/ -name '*.cpp')
MAINS=$(shell find $(MAIN_DIR)/ -name '*.cpp')
MAIN_OBJECTS=$(MAINS:$(MAIN_DIR)/%.cpp=$(OBJECT_DIR)/%.o)
OBJECTS=$(SOURCES:$(SOURCE_DIR)/%.cpp=$(OBJECT_DIR)/%.o)
EXECUTABLES=$(MAINS:$(MAIN_DIR)/%.cpp=%)

CC=g++
CFLAGS=-c -g -O3 -I$(SOURCE_DIR)/

all: $(EXECUTABLES)

.PHONY: clean test

.PRECIOUS: $(OBJECT_DIR)/%.o

$(EXECUTABLES): $(MAIN_OBJECTS) $(OBJECTS)
	$(CC) $(OBJECT_DIR)/$@.o $(OBJECTS) -o $(EXECUTABLE_DIR)/$@

$(OBJECT_DIR)/%.o: $(MAIN_DIR)/%.cpp
	$(CC) $(CFLAGS) $< -o $@

$(OBJECT_DIR)/%.o: $(SOURCE_DIR)/%.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(MAIN_OBJECTS) $(OBJECTS) $(EXECUTABLE_DIR)/*