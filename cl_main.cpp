#include<ctime>
#include <iostream>
#include <vector>

#include <boost/thread/thread.hpp>
#include <boost/random.hpp>

/*
  #include <boost/compute/source.hpp>
  #include <boost/compute/system.hpp>
  #include <boost/compute/algorithm/inclusive_scan.hpp>
  #include <boost/compute/algorithm/inclusive_scan.hpp>
  #include <boost/compute/random/mersenne_twister_engine.hpp>
  #include <boost/compute/random/uniform_real_distribution.hpp>
*/

#include <boost/compute/source.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/inclusive_scan.hpp>
#include <boost/compute/random/mersenne_twister_engine.hpp>
#include <boost/compute/random/default_random_engine.hpp>
#include <boost/compute/random/uniform_real_distribution.hpp>

//#include <boost/compute/event.hpp>
//#include <boost/compute/async/future.hpp>


namespace compute = boost::compute;
using compute::int2_;
using compute::char2_;

const int NUMBER_OF_PARTICLE = 16;
const int STEPS = NUMBER_OF_PARTICLE * 16;
const int TIMES = STEPS/NUMBER_OF_PARTICLE;


int main(){
  //for multi thread variables
  time_t start, end;

  time(&start);

  // get default device and setup context
  std::vector<compute::device> devices = compute::system::devices();

  {
    compute::device temp = devices[1];
    devices[1] = devices[2];
    devices[2] = temp;
  }

  //print devices
  for(int i = 0; i < devices.size(); ++i){
    std::cout << "id::" << i << " " << devices[i].name() << std::endl;
  }

  compute::context context(devices[1]);
  compute::command_queue queues(compute::command_queue(context, devices[1]));


  //random values
  compute::default_random_engine random_engine(queues);
  compute::uniform_real_distribution<float> random_distribution(0.f, 1.f);
  compute::vector<float> random_values(STEPS, context);
  compute::vector<int2_> particles(NUMBER_OF_PARTICLE, context);


  //compile move particle
  compute::program move_program = compute::program::create_with_source_file("./particle_move.cl", context);
  try {
    move_program.build();
  }catch(compute::opencl_error &e){
    std::cout << move_program.build_log() << std::endl;
    //std::cout << e << std::endl;
  }

  //make kernel for move
  compute::kernel move_kernel(move_program, "particle_move");
  move_kernel.set_arg(0, particles);
  move_kernel.set_arg(1, sizeof(int), &TIMES);
  move_kernel.set_arg(2, random_values);
  std::cout << TIMES << std::endl;



  size_t offset = 0;
  //size_t global_work_size = NUMBER_OF_PARTICLE;
  size_t global_work_size = NUMBER_OF_PARTICLE;
  size_t local_work_size = 1;
  for(int i = 0; i < 100; ++i){

    random_distribution.generate(random_values.begin(), random_values.end(), random_engine, queues);
    queues.enqueue_nd_range_kernel(move_kernel, 1, &offset, &global_work_size, &local_work_size);


    /*
    std::vector<float> result(STEPS);
    compute::copy(random_values.begin(), random_values.end(), result.begin(), queues);
    for(int j = 0; j < result.size(); j++)
      std::cout << j << ":: " <<result[j]  << std::endl;
    */

    // if(i % 1 == 0){
    //std::cout << i << std::endl;
    //}
   }


  std::vector<int2_> result(NUMBER_OF_PARTICLE);
  compute::copy(particles.begin(), particles.end(), result.begin(), queues);
  for(int i = 0; i < result.size(); i++)
    std::cout << result[i]  << std::endl;


  time(&end);
  std::cout << difftime(end, start) << std::endl;


  return 0;
}
