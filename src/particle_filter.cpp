/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;
	default_random_engine gen;

	normal_distribution<double> x_init(x, std[0]);
    normal_distribution<double> y_init(y, std[1]);
    normal_distribution<double> theta_init(theta, std[2]);

    for (int i = 0; i < num_particles; i++)
    {
        Particle p = {i+1, x_init(gen), y_init(gen), theta_init(gen), 1.0};
        particles.push_back(p);
        weights.push_back(1.0);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
    
    normal_distribution<double> x_noise(0, std_pos[0]);   
    normal_distribution<double> y_noise(0, std_pos[1]);
    normal_distribution<double> theta_noise(0, std_pos[2]);

    for (int i = 0; i < num_particles; i++)
    {
        if (fabs(yaw_rate) < 0.0001)
        {
            particles[i].x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
            particles[i].y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
            particles[i].theta = particles[i].theta;

        }
        else
        {
            particles[i].x = particles[i].x + velocity*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta))/yaw_rate;
            particles[i].y = particles[i].y + velocity*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t))/yaw_rate;
            particles[i].theta = particles[i].theta + yaw_rate*delta_t;
        }

        particles[i].x += x_noise(gen);
        particles[i].y += y_noise(gen);
        particles[i].theta += theta_noise(gen);
    }   

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++){
		double dist_cur = 1e5;
		int index_association;
		for (int j = 0; j < predicted.size(); j++){
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if (distance < dist_cur){
				dist_cur = distance;
				index_association = predicted[j].id;
			}
		}
		observations[i].id = index_association;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double gauss_norm = 1.0/(2.0*M_PI*std_landmark[0]*std_landmark[1]);

	for (int i = 0; i < num_particles; i++){
		vector<LandmarkObs> observations_trans(observations.size());
		for (int j = 0; j < observations.size(); j++)
        {
            observations_trans[j].x = particles[i].x + observations[j].x*cos(particles[i].theta) - observations[j].y*sin(particles[i].theta);
            observations_trans[j].y = particles[i].y + observations[j].x*sin(particles[i].theta) + observations[j].y*cos(particles[i].theta);
        }

        vector<LandmarkObs> landmarks_in_range;
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
        {
            Map::single_landmark_s landmark_j = map_landmarks.landmark_list[j];

            double distance = dist(particles[i].x, particles[i].y, landmark_j.x_f, landmark_j.y_f);

            if (distance < sensor_range)
            {
                LandmarkObs landmark_in_range = {landmark_j.id_i, landmark_j.x_f, landmark_j.y_f};
                landmarks_in_range.push_back(landmark_in_range);
            }
        }
        dataAssociation(landmarks_in_range, observations_trans);

        double weight_ = 1.0;
        for (int j = 0; j < observations_trans.size(); j++){
        	double x_pred;
        	double y_pred; 
        	for (int k = 0; k < landmarks_in_range.size(); k++){
        		if (landmarks_in_range[k].id == observations_trans[j].id){
        			x_pred = landmarks_in_range[k].x;
        			y_pred = landmarks_in_range[k].y;
        		}
        	}
        	weight_ *= gauss_norm * exp(-(pow((observations_trans[j].x - x_pred), 2)/
        	(2*pow(std_landmark[0], 2)) + pow(observations_trans[j].y - y_pred, 2)/
        	(2*pow(std_landmark[1], 2))));
        }
        particles[i].weight = weight_;
        weights[i] = weight_;

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> disc_dist(weights.begin(), weights.end());
	vector<Particle> particles_resample(num_particles);

	for (int i = 0; i < num_particles; i++){
		particles_resample[i] = particles[disc_dist(gen)];
	}
	particles = particles_resample;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
