/** 
Deep Neural Network algorithm, example file
Copyright (C) 2016 Stefan Ivanovic 

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>
**/



#include "StivDLNN.hpp"
using namespace std;


int main()
{
	vector<double> inputs; //Inputs vector
	vector<double> outputs; //Outputs vector
	int HidLayers = 3; //Number of hidden layers
	int HidNeurons = 4; //Number of neurons per hidden layer
	inputs.push_back(0.5); //Set input/s
	inputs.push_back(0.1);
	outputs.push_back(0.01); //Set output/s
	
	Net Net1(inputs,HidLayers,HidNeurons,outputs); //Create a Net
	
	int iterations = 100000; //Number of training repetitions
	int learning_speed = 2; //Speed at which the network learns. A higher speed means faster weight adjustment but less accuracy
		
	Net1.Train(iterations,learning_speed);
	
	string filename = "moki"; //File in which to save the trained network
	
	Net1.Save(filename); //Save network
	
	Net Net2(inputs,HidLayers,HidNeurons,outputs); //Create another Net to demonstrate how to load a net
	
	Net2.Set_weights(filename); //Load the saved network
	
	vector<double> Net2_outputs; //Run() function returns a vector of output values, thus we create a vector to hold them
	
	Net2_outputs = Net2.Run(); //Put outputs of the network into our created vector
	
	for(double i : Net2_outputs) //Print our results on the screen
	{
		cout << i << endl;
	}
	
	
	return 0;
}
