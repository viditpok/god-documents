# Authored by Christopher Tam for Georgia Tech's CS 2200
#Edited by Tanner Muldoon

TARGET = os-sim

CC     = gcc
CFLAGS = -Wall -Wextra -Wsign-conversion -Wpointer-arith -Wcast-qual -Wwrite-strings -Wshadow -Wmissing-prototypes -Wpedantic -Wwrite-strings -g -std=gnu99 -lm

LFLAGS = -lpthread

SRCDIR = src
SRCFILE = $(SRCDIR)/student.c
INCDIR = $(SRCDIR)
BINDIR = .

SUBMIT_FILES  = $(SRCFILE) Makefile answers.txt
SUBMISSION_NAME = project4-scheduling

SRC := $(wildcard $(SRCDIR)/*.c)
INC := $(wildcard $(INCDIR)/*.h)

INCFLAGS := $(patsubst %/,-I%,$(dir $(wildcard $(INCDIR)/.)))

.PHONY: all
all:
	@$(MAKE) release && \
	echo "$$(tput setaf 3)$$(tput bold)Note:$$(tput sgr0) this project compiled with release flags by default. To compile for debugging, please use $$(tput setaf 6)$$(tput bold)make debug$$(tput sgr0)."

.PHONY: debug
debug: CFLAGS += -ggdb -g3 -DDEBUG
debug: $(BINDIR)/$(TARGET)

.PHONY: tsan-debug
tsan-debug: CFLAGS += -fsanitize=thread
tsan-debug: debug

.PHONY: release
release: CFLAGS += -mtune=native -O2
release: $(BINDIR)/$(TARGET)

.PHONY: clean
clean:
	@rm -f $(BINDIR)/$(TARGET)
	@rm -rf $(BINDIR)/$(TARGET).dSYM

.PHONY: submit
submit:
	@cp $(SRCFILE) ./
	@(zip $(SUBMISSION_NAME).zip ./student.c ./answers.txt && \
	echo "Created submission archive $$(tput bold)$(SUBMISSION_NAME).zip$$(tput sgr0).")
	@rm ./student.c

$(BINDIR)/$(TARGET): $(SRC) $(INC)
	@mkdir -p $(BINDIR)
	@$(CC) $(CFLAGS) $(INCFLAGS) $(SRC) -o $@ $(LFLAGS)
