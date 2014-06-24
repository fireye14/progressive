#include <optix_world.h>
#include "helpers.h"
#include <math.h>

using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float  importance;
  int    depth;
};

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(float3, bad_color, , );
rtBuffer<uchar4, 2> output_buffer;
rtBuffer<float3, 2> queue_buffer;
rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, radiance_ray_type, , );

// change block_size to how big you want each "pixel" to be
// a value of 2 represents a 2x2 pixel
rtDeclareVariable(const unsigned int, block_size, , ) = 2;
rtDeclareVariable(const unsigned int, half_block_size, , ) = block_size >> 1;


//trace the ray through screen_coord
static __device__ __inline__ float3 trace( float2 screen_coord )
{
  size_t2 screen = output_buffer.size();
  float2 d = screen_coord / make_float2(screen) * 2.f - 1.f;
  float3 ray_origin = eye;
  float3 ray_direction = normalize(d.x*U + d.y*V + W);
  
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);
  return prd.result;
}

//find the average color of the block of pixels, then find variance of 
//color in the block of pixels and set the pixel color to that color
static __device__ __inline__ void avgColor( const uint2& index )
{
  size_t2 screen = output_buffer.size();         //size of the screen
  float3 color = make_float3(0);                 //keeps track of the colors at each pixel in the block
  float variance_red = 0;						 //variance in the reds
  float variance_green = 0;						 //variance in the greens
  float variance_blue = 0;						 //variance in the blues
  float variance_total = 0;					     //total variance of the colors

  //make sure we look at pixels that are on the screen
  unsigned int min_x = max( index.x-half_block_size, 0u );
  unsigned int max_x = min( index.x+half_block_size, (unsigned int) screen.x );
  unsigned int min_y = max( index.y-half_block_size, 0u );
  unsigned int max_y = min( index.y+half_block_size, (unsigned int) screen.y );
  
  for ( unsigned int i = min_x; i < max_x; ++i ) {
    for ( unsigned int j = min_y; j < max_y; ++j ) {
        //keep a running total of the color values
		queue_buffer[make_uint2(i,j)] = trace(make_float2(i, j));
		color += queue_buffer[make_uint2(i,j)];
    }
  }

  //calculate the average color of the block of pixels 
  color /= (block_size * block_size);

  for ( unsigned int i = min_x; i < max_x; ++i ) {
    for ( unsigned int j = min_y; j < max_y; ++j ) {
	  //variance sum += (sample color - average color)^2      for each sample
	  //calculate variance of each color
	  float3 temp = queue_buffer[make_uint2(i,j)];

	  variance_red += (temp.x - color.x) * (temp.x - color.x);
      variance_green += (temp.y - color.y) * (temp.y - color.y);
      variance_blue += (temp.z - color.z) * (temp.z - color.z);
    }
  }

  //variance = variance sum / num samples
  //standard deviation = sqrtf(variance)
  //multiplied to accentuate colors more
  variance_red = sqrtf((variance_red / (block_size * block_size)))*2;// * 8;
  variance_green = sqrtf((variance_green / (block_size * block_size)))*2;// * 8;
  variance_blue = sqrtf((variance_blue / (block_size * block_size)))*2;// * 8;

  //calculate the luminance value of the three variances
  variance_total = optix::luminance(make_float3(variance_red, variance_green, variance_blue));

  for ( unsigned int i = min_x; i < max_x; ++i ) {
    for ( unsigned int j = min_y; j < max_y; ++j ) {
	  //set the color of all the pixels in the block to the amount of variance among them
	  output_buffer[make_uint2(i, j)] = make_color(make_float3(variance_total));
	}
  }
}


//check whether the pixel is in the center of block
static __device__ __inline__ bool shouldTrace( const uint2& index )
{
  uint2        shifted_index = make_uint2( index.x + half_block_size, index.y + half_block_size ); 
  size_t2      screen        = output_buffer.size(); 
  return ( shifted_index.x % block_size == 0 && shifted_index.y % block_size == 0 ) ||
         ( index.x == screen.x-1 && screen.x % block_size <= half_block_size && shifted_index.y % block_size == 0 ) ||
         ( index.y == screen.y-1 && screen.y % block_size <= half_block_size && shifted_index.x % block_size == 0 );
}


RT_PROGRAM void pinhole_camera()
{
  //check whether or not the current index is at the center of a block
  if(shouldTrace(launch_index)) 
  {
	  avgColor(launch_index);
  }
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_color(bad_color);
}