#CC = g++
#CFLAGS = -c
#SOURCES =  ParticleSwarm.cpp OP.cpp Rastriging.cpp Particle.cpp Swarm.cpp
#OBJECTS = $(SOURCES:.cpp=.o)
#EXECUTABLE = ParticleSwarm

#all: $(OBJECTS) $(EXECUTABLE)

#$(EXECUTABLE) : $(OBJECTS)
#		$(CC) $(OBJECTS) -o $@  

#.cpp.o: *.h
#	$(CC) $(CFLAGS) $< -o $@

OBJECTS = ParticleSwarm.o OP.o Rastriging.o Particle.o Swarm.o
LINKEROPTIONS = -lm -lGL
NVCCOPTIONS = -arch sm_75
GXXFLAGS = -std=c++11

program : $(OBJECTS)
nvcc $(OBJECTS) -o program $(LINKEROPTIONS)
OP.o : OP.cpp OP.h
Rastriging.o : Rastriging.cpp Rastriging.h
Particle.o : Particle.cpp Particle.h
g++ -c $(GXXFLAGS)
ParticleSwarm.o : ParticleSwarm.cu 
Swarm.o :Swarm.cu Swarm.h
nvcc -c cuda_code.cu $(NVCCOPTIONS)

clean :
	-rm -f $(OBJECTS) 

.PHONY: all clean