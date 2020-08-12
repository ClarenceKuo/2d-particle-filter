/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#define DEBUG 0

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  //Init materials
  std::default_random_engine gen;
  double std_x, std_y, std_theta;
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  //Set the number of particles.
  num_particles = 5;
  
  //Initialize all particles to first position with Gaussian noise and wieght = 1
  for(int i = 0; i < num_particles; i ++){
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    
    //Push back to public attribute vectors: particles
    particles.push_back(p);
	}
  
  //init completed
  is_initialized = 1;
  if(DEBUG){
    for(int i = 0; i < num_particles; i ++){
    std::cout << "Sample " << i + 1 << " " << particles[i].x << " " << particles[i].y << " " << particles[i].theta << " " << particles[i].weight << std::endl;
  	}
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  std::default_random_engine gen;
  
  //prevent div_zero 
  if (yaw_rate == 0){
  	yaw_rate = 0.000001;
  }
  for(int i = 0; i < num_particles; i ++){
    Particle *p = &particles[i];
    p->x += (velocity/yaw_rate)*( sin(p->theta + yaw_rate * delta_t ) - sin(p->theta));
    p->y += (velocity/yaw_rate)*( cos(p->theta) - cos(p->theta + yaw_rate * delta_t ));
    p->theta += yaw_rate * delta_t;
    
    normal_distribution<double> dist_x(p->x, std_pos[0]);
    normal_distribution<double> dist_y(p->y, std_pos[1]);
    normal_distribution<double> dist_theta(p->theta, std_pos[2]);
    
    //Prediction
	p->x = dist_x(gen);
    p->y = dist_y(gen);
    p->theta = dist_theta(gen);
    
    if(DEBUG){
      std::cout << "Predicted location: " << particles[i].x << particles[i].y << std::endl;
  	}
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  double weights_sum = 0;
  for(int i=0; i<num_particles; ++i){
    Particle *p = &particles[i];
    double wt = 1.0;

    // convert obs: vehicle -> map
    for(LandmarkObs current_obs : observations){
      LandmarkObs transformed_obs;

      transformed_obs.x = (current_obs.x * cos(p->theta)) - (current_obs.y * sin(p->theta)) + p->x;
      transformed_obs.y = (current_obs.x * sin(p->theta)) + (current_obs.y * cos(p->theta)) + p->y;
      transformed_obs.id = current_obs.id;

      // find the closest observation and assign to the landmark
      Map::single_landmark_s landmark;
      double distance_min = std::numeric_limits<double>::max();

      for(Map::single_landmark_s location : map_landmarks.landmark_list){
        double distance = dist(transformed_obs.x, transformed_obs.y, location.x_f, location.y_f);
        if(distance < distance_min){
          distance_min = distance;
          landmark = location;
        }
      }
      
      // update weights
      double num = exp(-0.5 * (pow((transformed_obs.x - landmark.x_f), 2) / pow(std_x, 2) + pow((transformed_obs.y - landmark.y_f), 2) / pow(std_y, 2)));
      double denom = 2 * M_PI * std_x * std_y;
      wt *= num/denom;
    }
        weights_sum += wt;
        p->weight = wt;
  }
  // normalize
  for(int i = 0; i < num_particles; i ++){
    Particle *p = &particles[i];
    p->weight /= weights_sum;
  }
}

void ParticleFilter::resample() {
  std::default_random_engine gen;
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  vector<Particle> resampled_particles;
  
  // Using Sebestian's mysterious resampling wheel
  double max_weight = 0.0;
  double beta = 0.0;
  double rand1 = unif(gen);
  int index = (int)(rand1 * num_particles);
  for(int i = 0; i < num_particles; i ++){
    Particle *p = &particles[i];
    if (p->weight > max_weight)
      max_weight = p->weight;
  }
  for (int i = 0; i < num_particles; i ++){
    double rand2 = unif(gen);
    beta += rand2 * 2 * max_weight;
    while(beta > particles[index].weight){
      beta -= particles[index].weight;
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }
    particles = resampled_particles;
}


void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}