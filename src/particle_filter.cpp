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
#include <initializer_list>

#include "particle_filter.h"

using namespace std;

static random_device rd;
static mt19937 gen(rd());

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	num_particles = 500;
	normal_distribution<double> x1(x, std[0]);
	normal_distribution<double> y1(y, std[1]);
	normal_distribution<double> theta1(theta, std[2]);
	for(int i=0; i<num_particles; i++) {
		Particle r;
		r.id = i;
		r.x = x1(gen);
		r.y = y1(gen);
		r.theta = theta1(gen);
		r.weight = 1.;
		particles.push_back(r);
		weights.push_back(1.);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	normal_distribution<double> x1(0., std_pos[0]);
	normal_distribution<double> y1(0., std_pos[1]);
	normal_distribution<double> theta1(0., std_pos[2]);
	for(int i=0; i<num_particles; i++) {
		if(fabs(yaw_rate)<0.001) {
			double dist = velocity*delta_t;
			particles[i].x += dist*cos(particles[i].theta);
			particles[i].y += dist*sin(particles[i].theta);
		} else {
			double theta_old = particles[i].theta;
			particles[i].theta += yaw_rate*delta_t;
			particles[i].x += velocity/yaw_rate*(sin(particles[i].theta)-sin(theta_old));
			particles[i].y += velocity/yaw_rate*(cos(theta_old)-cos(particles[i].theta));
		}
		particles[i].x += x1(gen);
		particles[i].y += y1(gen);
		particles[i].theta += theta1(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//   Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	vector<LandmarkObs> obs_buf;
	for(unsigned int i=0; i<predicted.size(); i++) {
		double dist_min = 10000.;
		unsigned int k_min = 0;
		for(unsigned int k=0; k<observations.size(); k++) {
			double diff_x = predicted[i].x - observations[k].x;
			double diff_y = predicted[i].y - observations[k].y;
			double dist_test = sqrt(pow(diff_x,2)+pow(diff_y,2));
			if(dist_test < dist_min) {
				dist_min = dist_test;
				k_min = k;
			}
		}
		obs_buf.push_back(observations[k_min]);
	}
	observations = obs_buf;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	//   Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for(int i=0; i<num_particles; i++) {
		vector<LandmarkObs> predicted_measurements;
		for(unsigned int k=0; k<map_landmarks.landmark_list.size(); k++) {
			LandmarkObs pred;
			//compute measurement predictions in earthfixed coordinates
			double E_x = map_landmarks.landmark_list[k].x_f - particles[i].x;
			double E_y = map_landmarks.landmark_list[k].y_f - particles[i].y;
			if(sqrt(pow(E_x,2)+pow(E_y,2)) < sensor_range) {
				pred.id =  k;
				//transform to vehicle coordinates
				pred.x  =  E_x*cos(particles[i].theta)+E_y*sin(particles[i].theta);
				pred.y  = -E_x*sin(particles[i].theta)+E_y*cos(particles[i].theta);
				predicted_measurements.push_back(pred);
			}
		}

		if(!predicted_measurements.empty() && (predicted_measurements.size() == observations.size())) {
			//associate measurements and predictions
			vector<LandmarkObs> obs = observations;
			dataAssociation(predicted_measurements, obs);

			//compute new particle weight
			particles[i].weight = 1.;
			for(unsigned int k=0; k<obs.size(); k++) {
				double e_x = obs[k].x - predicted_measurements[k].x;
				double e_y = obs[k].y - predicted_measurements[k].y;
				particles[i].weight *= exp(-((pow(e_x,2)/(2.*pow(std_landmark[0],2))) + (pow(e_y,2)/(2.*pow(std_landmark[1],2))))) / (2.*M_PI*std_landmark[0]*std_landmark[1]);
				weights[i] = particles[i].weight;
			}
		} else {
			particles[i].weight = 0.;
			weights[i] = 0.;
		}
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> pbuf;
	discrete_distribution<unsigned int> d(weights.begin(),weights.end());
	for(unsigned int i=0; i<num_particles; i++) {
		pbuf.push_back(particles[d(gen)]);
	}
	particles = pbuf;
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
