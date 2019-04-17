OBJECTS = ParticleSwarm.o OP.o Rastriging.o Particle.o Swarm.o
LINKEROPTIONS = -lm -lGL
NVCCOPTIONS = -arch sm_75
GXXFLAGS = -std=c++11

program : $(OBJECTS)
	nvcc $(OBJECTS) -o ParticleSwarm $(LINKEROPTIONS)
OP.o : OP.cpp OP.h
	g++ -c OP.cpp
Rastriging.o : Rastriging.cpp Rastriging.h
	g++ -c Rastriging.cpp
Particle.o : Particle.cpp Particle.h
	g++ -c Particle.cpp
ParticleSwarm.o : ParticleSwarm.cu 
	nvcc -c ParticleSwarm.cu $(NVCCOPTIONS)
Swarm.o : Swarm.cu Swarm.h
	nvcc -c Swarm.cu $(NVCCOPTIONS)

clean :
	-rm -f $(OBJECTS) 

.PHONY: all clean