__kernel void particle_move(__global int2* particles, const int times, __global float* random){
    
    const int WIDTH = 50;
    const int HEIGHT = WIDTH * 2;
    
    //__local const int  _index[] = {0, 4, 1, 2, 3};
    
    int gid = get_global_id(0);
    //int lid = get_local_id(0);
    //printf("%d\t%d\n", gid, lid);
    
    int i;
    
    const int upDown[]    = {0, -1, 0, 1,  0};
    const int rightLeft[] = {0,  0, 1, 0, -1};
    const int jumpIndex[] = {0,  4, 1, 2,  3};
    
    char index;
    float xi;
    int2 temp;
    
    for(i = 0; i < times; ++i){
        xi = 4 * random[i * gid + i] + 1;
        index = (char)xi;
        
        //move
        temp[0] = particles[i * gid + i].x + upDown[index];
        temp[1] = particles[i * gid + i].y + rightLeft[index];
        index = ((temp[0] >= 0 && temp[0] < HEIGHT) && (temp[1] >= 0 && temp[1] < WIDTH))? index : 0;
        if(index != 0){
            particles[i * gid + i] = temp;
        }
        
        //jump
        if(xi - (int)xi <= 0.5f){
            index = jumpIndex[index];
            temp[0] = particles[i * gid + i].x + upDown[index];
            temp[1] = particles[i * gid + i].y + rightLeft[index];
            index = ((temp[0] >= 0 && temp[0] < HEIGHT) && (temp[1] >= 0 && temp[1] < WIDTH))? index : 0;
            
            if(index != 0){
                particles[i * gid + i] = temp;
            }
        }
        
        if(xi == 1){
            printf("_A_\n");
        }
        //printf("times:%d  id:%d %f\n", i, gid, random[i]);
    }
}