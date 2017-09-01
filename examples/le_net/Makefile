ROOT:=../../../Mali_OpenCL_SDK

include $(ROOT)/platform.mk

CFLAGS:=-c -Wall -I$(ROOT)/include -I$(ROOT)/common -I.

LDFLAGS:=-L$(ROOT)/lib -L$(ROOT)/common -lOpenCL -lCommon

SOURCES:=le_net.cpp
HEADERS:=$(ROOT)/common/common.h $(ROOT)/common/image.h

OBJECTS:=$(SOURCES:.cpp=.o)

EXECUTABLE:=le_net

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) libOpenCL libCommon
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

$(OBJECTS): $(HEADERS)

install: $(EXECUTABLE)
	-$(MKDIR) "$(ROOT)/bin/$(EXECUTABLE)/assets"
	$(CP) "$(EXECUTABLE)" "$(ROOT)/bin/$(EXECUTABLE)/$(EXECUTABLE)"
	cd assets $(CONCATENATE) $(CP) * "../$(ROOT)/bin/$(EXECUTABLE)/assets/"

.PHONY: clean libOpenCL libCommon

clean:
	$(RM) $(OBJECTS) $(EXECUTABLE)

libOpenCL:
	cd $(ROOT)/lib $(CONCATENATE) $(MAKE) libOpenCL.so

libCommon:
	cd $(ROOT)/common/ $(CONCATENATE) $(MAKE) libCommon.a
