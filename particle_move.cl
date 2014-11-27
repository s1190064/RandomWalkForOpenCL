__kernel void particle_move(__global int2* particles, __global  char2* index_info, const int times, int num_particle, __global int2* move){
    
    const int WIDTH = 50;
    const int HEIGHT = WIDTH * 2;
    
    //__local const int  _index[] = {0, 4, 1, 2, 3};
    
    int gid = get_global_id(0);
    //int lid = get_local_id(0);
    //printf("%d\t%d\n", gid, lid);
    
    int i, j;
    int2 temp;
    char2 index;
    int ind;
    
    for(i = 20*gid; i < 20*gid + 2; ++i){
        for(j = 0; j < times; ++j){
            
            index = (char2)index_info[i * times + j];
            
            //move
            
            temp = (int2)particles[i] + (int2)move[(int)index[0]];
            ind = ((temp[0] >= 0 && temp[0] < HEIGHT) && (temp[1] >= 0 && temp[1] < WIDTH))? (int)index[0] : 0;
            
            particles[i] += (int2)move[ind];
            
            
            //jump
            temp = (int2)particles[i] + (int2)move[(int)index[1]];
            ind = ((temp[0] >= 0 && temp[0] < HEIGHT) && (temp[1] >= 0 && temp[1] < WIDTH))? (int)index[1] : 0;
            
            particles[i] += (int2)move[ind];
        }
    }
     
}
