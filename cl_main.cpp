#include<ctime>
#include <iostream>
#include <vector>

#include <boost/thread/thread.hpp>
#include <boost/random.hpp>

#include <boost/compute/source.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/inclusive_scan.hpp>
#include <boost/compute/algorithm/inclusive_scan.hpp>
#include <boost/compute/random/mersenne_twister_engine.hpp>
#include <boost/compute/random/uniform_real_distribution.hpp>

#include <boost/compute/event.hpp>
#include <boost/compute/async/future.hpp>


#include "constant_variables.hpp"
//#include "source.hpp"

namespace compute = boost::compute;
using compute::int2_;
using compute::char2_;

typedef boost::random::mt19937 MersenneTwister;
typedef boost::random::uniform_real_distribution<> UniformRealDistribution;
typedef boost::random::variate_generator< MersenneTwister, UniformRealDistribution> Random;

MersenneTwister **mt;
UniformRealDistribution **range;
Random **randomValue;
//float *randomArray;
std::vector<float> randomArray(NUMBER_OF_PARTICLE * ONE_TIME);


std::vector<int2_> particle(NUMBER_OF_PARTICLE);
std::vector<compute::vector<int2_> > particle_cl;
std::vector<char2_> k_index(NUMBER_OF_PARTICLE * ONE_TIME);



void initialzeGrobalVariables(const int numberOfThreads){

  //randomArray = new float[NUMBER_OF_PARTICLE];

  //initialize random values
  mt = new MersenneTwister*[numberOfThreads];
  range = new UniformRealDistribution*[numberOfThreads];
  randomValue = new Random*[numberOfThreads];

  for(int i = 0; i < numberOfThreads; ++i){
    mt[i] = new MersenneTwister(static_cast<unsigned long>(time(0)) + i);
    //range[i] = new UniformRealDistribution(0,1);
    range[i] = new UniformRealDistribution(0, 4);
    randomValue[i] = new Random(*(mt[i]), *(range[i]));
  }
}


void deleteGrobalVariables(const int numberOfThreads){
  for(int i = 0; i < numberOfThreads; ++i){
    delete mt[i];
    delete range[i];
    delete randomValue[i];
  }

  delete[] mt;
  delete[] range;
  delete[] randomValue;

  //delete[] randomArray;
}




void do_work(const int P, const int start, const int end){
  for(int i = start; i < end; ++i){
    randomArray[i] = (*(randomValue[P]))();
  }
}


void do_work2(const int P, const int start, const int end){
  float p;
  const int _index[] = {0, 4, 1, 2, 3};

  for(int i = start; i < end; ++i){
    p = randomArray[i];
    char2_ temp = k_index[i];
    //k_index[i] = (char2_)(p + 1.f, (p - (int)p >= 0.5f)? _index[(int)p+1] : 0);
    temp[0] = p + 1.f;
    temp[1] = (p - (int)p >= 0.5f)? _index[(int)p+1] : 0;

    k_index[i] = (char2_)temp;
  }
}


void makeRandomArray(const int P){
  std::vector<boost::thread *> threads(P);

  //create threads
  for (int j = 0; j < P; ++j){
    int section = randomArray.size()/P;
    threads[j] = (new boost::thread(do_work, j, j*section, (j+1) * section));
  }

  //join and delete
  for (int j = 0; j < P; ++j){
    threads[j]->join();
    delete threads[j];
  }
}


void makeIndex(const int P){
  std::vector<boost::thread *> threads(P);

  //create threads
  for (int j = 0; j < P; ++j){
    int section = k_index.size()/P;
    threads[j] = (new boost::thread(do_work2, j, j*section, (j+1) * section));
  }

  //join and delete
  for (int j = 0; j < P; ++j){
    threads[j]->join();
    delete threads[j];
  }
}


int main(){
  //for multi thread variables
  const int P = boost::thread::hardware_concurrency();
  time_t start, end;

  time(&start);
  initialzeGrobalVariables(P);

  // get default device and setup context
  std::vector<compute::device> devices = compute::system::devices();

  {
    compute::device temp = devices[1];
    devices[1] = devices[2];
    devices[2] = temp;
  }
    /*
    {
        compute::device temp = devices[0];
        devices[1] = devices[0];
        devices[0] = temp;
    }
     */
  //print devices
  for(int i = 0; i < devices.size(); ++i){
    std::cout << "id::" << i << " " << devices[i].name() << std::endl;
  }


  std::vector<compute::context> contexts;
  contexts.push_back(compute::context(devices[1]));
  //contexts.push_back(compute::context(devices[2]));


  std::vector<compute::command_queue> queues;
  queues.push_back(compute::command_queue(contexts[0], devices[1]));




  //===============
  //MAIN 
  //===============
  std::vector<compute::vector<float> > random_values;
  std::vector<compute::vector<char2_> > index_cl;

  std::vector<int2_> move;
  compute::vector<int2_> move_cl(5, contexts[0]);
  {
    int2_ temp(0, 0);
    move.push_back(temp);

    temp[0] = -1; temp[1] = 0;
    move.push_back(temp);

    temp[0] = 0; temp[1] = 1;
    move.push_back(temp);

    temp[0] = 1; temp[1] = 0;
    move.push_back(temp);

    temp[0] = 0; temp[1] = -1;
    move.push_back(temp);
  }

  compute::copy(move.begin(), move.end(), move_cl.begin(), queues[0]);

  index_cl.push_back(compute::vector<char2_>(NUMBER_OF_PARTICLE * ONE_TIME, contexts[0]));

  for(int i = 0; i < contexts.size(); ++i){
    random_values.push_back(compute::vector<float>(NUMBER_OF_PARTICLE, contexts[i]));
    particle_cl.push_back(compute::vector<int2_>(NUMBER_OF_PARTICLE, contexts[i]));
  }


  for(int i = 0; i < particle.size(); ++i){
    int2_ temp(0, 0);
    particle[i]= (int2_)temp;
  }


  compute::copy(particle.begin(), particle.end(), particle_cl[0].begin(), queues[0]);
  //compute::copy(particle.begin() + NUMBER_OF_PARTICLE/2, particle.end(), particle_cl[1].begin(), queues[1]);
  //compute::copy(particle.begin(), particle.end(), particle_cl.begin(), queues[0]);


  //make kernel
  //compute::program move_program = compute::program::create_with_source_file("./particle_move.cl", contexts[0]);
  compute::program move_program = compute::program::create_with_source_file("./particle_move.cl", contexts[0]);
  try {
    move_program.build();
  }
  catch(boost::compute::opencl_error &e){
    std::cout << "ERROR!!" <<std::endl;
    std::cout << move_program.build_log() << std::endl;
  }

  compute::kernel move_kernel(move_program, "particle_move");

 
  //move particle
  std::vector<float> rand(NUMBER_OF_PARTICLE/2);
  for(int i = 0; i < TIMES; i++){
    for(long int j = 0; j < TIME_OF_TRAJECTORY; j += ONE_TIME){
      //get random values
      makeRandomArray(P);
      makeIndex(P);

      //std::cout << "Start moving particle on GPU" << std::endl;

      //set index move particle
      compute::copy(k_index.begin(), k_index.end(), index_cl[0].begin(), queues[0]);
      if(j == 0){
      move_kernel.set_arg(0, particle_cl[0]);
      move_kernel.set_arg(2, sizeof(int), &ONE_TIME);
      move_kernel.set_arg(3, sizeof(int), &NUMBER_OF_PARTICLE);
      move_kernel.set_arg(4, move_cl);
      }
      move_kernel.set_arg(1, index_cl[0]);

      size_t offset = 0;
      //size_t global_work_size = NUMBER_OF_PARTICLE;
      size_t global_work_size = 60;
      size_t local_work_size = 1;

      queues[0].enqueue_nd_range_kernel(move_kernel, 1, &offset, &global_work_size, &local_work_size);
      //queues[0].finish();
      //compute::wait_list event(compute::event());
      //queues[0].enqueue_task(move_kernel, NULL);

      if(j % (ONE_TIME) == 0){
          std::cout << "K::" << j << " : "<< (double)(1.0*j)/(double)TIME_OF_TRAJECTORY * 100 << "%" << std::endl;
      }

    }
  }
  time(&end);

  //compute::copy(moveIndex_cl.begin(), moveIndex_cl.end(), particle.begin(), queue);
  
  compute::copy(particle_cl[0].begin(), particle_cl[0].end(), particle.begin(), queues[0]);
  //compute::copy(particle_cl[1].begin(), particle_cl[1].end(), particle.begin(), queues[1]);
    
  for(int i = 0; i < particle.size(); i++)
    std::cout << particle[i]  << std::endl;
  
  
  std::cout << "clock:: " << difftime(end, start) <<std::endl;
  deleteGrobalVariables(P);


  return 0;
}
