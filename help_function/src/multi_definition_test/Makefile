CC := icpx -fsycl

LD := $(CC)

#DPCT2001:0: You can link with more library by add them here.
LIB := 



ifeq ($(shell which icpx),)
    $(error ERROR - icpx compiler not found)
endif

ROOT_DIR     := $(shell dirname $(shell which icpx))
INCLUDE_SYCL := $(ROOT_DIR)/../include
INCLUDE_CL   := $(ROOT_DIR)/../include/sycl

FLAGS := -I${INCLUDE_SYCL} -I${INCLUDE_CL}


TARGET_0_SRC_0 = ./main.cpp
TARGET_0_OBJ_0 = ./main.o
TARGET_0_FLAG_0 = ${FLAGS}

TARGET_0_SRC_1 = ./foo.cpp
TARGET_0_OBJ_1 = ./foo.o
TARGET_0_FLAG_1 = ${FLAGS}

TARGET_0 := runfile

TARGET :=  ${TARGET_0}

.PHONY:all clean
OBJS_0 :=  ${TARGET_0_OBJ_0} ${TARGET_0_OBJ_1}
all: $(TARGET)
$(TARGET_0): $(OBJS_0)
	$(CC) -o $@ $^ $(LIB) 

$(TARGET_0_OBJ_0):$(TARGET_0_SRC_0)
	$(CC) -c ${TARGET_0_SRC_0} -o ${TARGET_0_OBJ_0} $(TARGET_0_FLAG_0)

$(TARGET_0_OBJ_1):$(TARGET_0_SRC_1)
	$(CC) -c ${TARGET_0_SRC_1} -o ${TARGET_0_OBJ_1} $(TARGET_0_FLAG_1)

clean:
	rm -f  ${OBJS_0} $(TARGET)
