# Authored by Christopher Tam for Georgia Tech's CS 2200
TARGET = rtp-client

CC     = gcc
CFLAGS = -Wall -Wextra -Wsign-conversion -Wpointer-arith -Wcast-qual -Wwrite-strings -Wshadow -Wmissing-prototypes -Wpedantic -Wwrite-strings -g -std=gnu99

LFLAGS = -lpthread

SRCDIR = src
INCDIR = $(SRCDIR)
BINDIR = .

SUBMIT_SUFFIX = -networking
SUBMIT_FILES  = $(SRC) $(INC) Makefile rtp-server.py
SUBMISSION_NAME = project5-networking

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

.PHONY: release
release: CFLAGS += -mtune=native -O2
release: $(BINDIR)/$(TARGET)

.PHONY: clean
clean:
	@rm -f $(BINDIR)/$(TARGET)
	@rm -rf $(BINDIR)/$(TARGET).dSYM

.PHONY: check-username
check-username:
	@if [ -z "$(GT_USERNAME)" ]; then \
		echo "Before running 'make submit', please set your GT Username in the environment"; \
		echo "Run the following to set your username: \"export GT_USERNAME=<your username>\""; \
		exit 1; \
	fi

.PHONY: submit
submit:
	@cp ./src/rtp.c ./
	@cp ./src/rtp.h ./
	@(zip $(SUBMISSION_NAME).zip ./rtp.c ./rtp.h && \
	echo "Created submission archive $$(tput bold)$(SUBMISSION_NAME).zip$$(tput sgr0).")
	@rm ./rtp.c ./rtp.h

$(BINDIR)/$(TARGET): $(SRC) $(INC)
	@mkdir -p $(BINDIR)
	@$(CC) $(CFLAGS) $(INCFLAGS) $(SRC) -o $@ $(LFLAGS)
