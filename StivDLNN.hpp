#include <iostream>
#include <math.h>
#include <vector>
#include <stdio.h>      
#include <stdlib.h>     
#include <time.h> 
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>

const double e = 2.71828182845904523536028747135266249775724709369995;

using namespace std;
class Neuron
{
	public:
	double a;
	double a_prime;
	double z;
	vector<double> weights;
	Neuron(int x);
	void Sigmoid(vector<double> x);
	double Cost(double desired_out);
	
};

Neuron::Neuron(int x)
{
   for (int i=0;i<x;i++)//Gives weights to the neuron based on how many neurons the prev layer has
   {
   	    double gi = (double) rand()/(double)(RAND_MAX);
		weights.push_back(gi);
   }
}
void Neuron::Sigmoid(vector<double> x)//Sigmoid function 
{
	int i;
	z=0;
	for(i=0;i<x.size();i++)
	{
		z += x.at(i)*weights.at(i);
	}
	a = 1/(1+pow(e,-z));
	a_prime = a * (1-a);
}

double Neuron::Cost(double desired_out)//Cost function
{
	return pow(desired_out - a,2)/2;
}


class Net//Neural Network
{
	public:
	vector<double> inputs;//Inputs
	vector< vector<Neuron> > hiddenLayers;//Neurons in each hidden layer
	vector< vector<double> > hiddenLayer_outputs;//Outputs from each hidden layer
	vector<double> hid_outputs;//Outputs from hidden layer,inputs to output layer
	vector<double> out_outputs;//Outputs from output layer
	vector<double> outputs;//Desired outputs
	vector<Neuron> hid_neurons;//Neurons in hidden layer
	vector<Neuron> out_neurons;//Neurons in output layer
	Net(vector<double> in, int hidLayers, int hidLayer_size, vector<double> out);//Constructor
	void Forward_Prop();//Forward propagate
	void Back_Prop(double speed);//Backward propagate
	void Save(string name);//Save weight values
	void Set_weights(string name);//Set weights from saved file
	void Train(int iter, int speed);//Trains the network
	vector<double> Run();
};

Net::Net(vector<double> in, int hidLayers, int hidLayer_size, vector<double> out)
{   //Constructor stores inputs,output,initial weights,Neurons into vectors
	int i,j;
	for (i=0;i<hidLayer_size;i++)
	{
		Neuron N(in.size());
		hid_neurons.push_back(N);
    }
    hiddenLayers.push_back(hid_neurons);
    hid_neurons.clear();
	for (j=1;j<hidLayers;j++)
	{
	   for (i=0;i<hidLayer_size;i++)
	   {
		   Neuron N(hidLayer_size);
		   hid_neurons.push_back(N);
       }
       hiddenLayers.push_back(hid_neurons);
       hid_neurons.clear();
    }
	for (i=0;i<out.size();i++)
	{
		Neuron N(hidLayer_size);
		out_neurons.push_back(N);
	}
	for(double i : in)
	{
		inputs.push_back(i);
	}
	for(double i : out)
	{
		outputs.push_back(i);
	}
}//End of func


void Net::Forward_Prop()//Forward Propagate
{
	out_outputs.clear();
	int i,j;
	for(Neuron &j : hiddenLayers.at(0))
	{
		j.Sigmoid(inputs);
		hid_outputs.push_back(j.a);
	}
	hiddenLayer_outputs.push_back(hid_outputs);
	hid_outputs.clear();
	for (i = 1; i< hiddenLayers.size();i++)
	{
	   for(Neuron &j : hiddenLayers.at(i))
	   {
	   	  j.Sigmoid(hiddenLayer_outputs.at(i-1));
		  hid_outputs.push_back(j.a);
	   }
	   hiddenLayer_outputs.push_back(hid_outputs);
	   hid_outputs.clear();
    }
	for(Neuron &k : out_neurons)
	{
		k.Sigmoid(hiddenLayer_outputs.at(hiddenLayer_outputs.size()-1));
		out_outputs.push_back(k.a);
	}
}//End of func


void Net::Back_Prop(double speed)//Backward Propagate
{
	int i;
	int j = 0;
	int k = 0;
	vector<double> deltas;
	vector<double> deltas2;
	double delta;
	
	for(i=0;i<hiddenLayers[0].size();i++)
	{
		deltas.push_back(0);
		deltas2.push_back(0);
	}

	for(Neuron &o : out_neurons)
	{
		delta = o.a_prime * o.Cost(outputs[j]);
		for(i=0;i<o.weights.size();i++)
		{
			o.weights[i] -= speed * delta * hiddenLayers[hiddenLayers.size()-1].at(i).a;
			deltas[i] += delta * o.weights[i];
		}
		j++;
	}

    if(hiddenLayers.size() > 1)
    {
	    for(j=0;j<hiddenLayers.size()-1;j++)
	    {
		    k=0;
		    for(Neuron &h : hiddenLayers[hiddenLayers.size()-(j+1)])
		    {
			    delta = h.a_prime * deltas[k];
			    for(i=0;i<h.weights.size();i++)
		        {
			        h.weights[i] -= speed * delta * hiddenLayers[hiddenLayers.size()-(j+2)].at(i).a;
			        deltas2[i] += delta * h.weights[i];
		        }
		        k++;
		    }
		    for (i=0;i<deltas.size();i++)
		    {
			    deltas[i] = deltas2[i];
			    deltas2[i] = 0;
		    }
	    }
    }

	
	k=0;
	for(Neuron &h1 : hiddenLayers[0])
	{
		delta = h1.a_prime * deltas[k];
		for(i=0;i<h1.weights.size();i++)
		{
			h1.weights[i] -= speed * delta * inputs[i];
		}
		k++;
	}	
}//End of func


void Net::Set_weights(string name)
{
	int i,j;
	int k = 0;
	string full_name = name + "_weights.txt";
	string line;
	vector<double> weights;
	ifstream file(full_name);
	
	while ( getline (file,line) )
    {
       weights.push_back(stod(line));
    }
    
    for(Neuron &h : hiddenLayers[0])
	{
		for(i=0;i<h.weights.size();i++)
		{
			h.weights[i] = weights[k];
			k++;
		}
	}
	
	for(j=1;j<hiddenLayers.size();j++)
	{
        for(Neuron &h : hiddenLayers[j])
		{
			for(i=0;i<h.weights.size();i++)
		    {
			    h.weights[i] = weights[k];
			    k++;
		    }
		}
	}
	
	for(Neuron &o : out_neurons)
	{
		for(i=0;i<o.weights.size();i++)
		{
			o.weights[i] = weights[k];
			k++;
		}
	} 
	file.close();
}//End of func


void Net::Save(string name)
{
	int i,j;
	string full_name;
	ofstream file;
    full_name = name + "_weights.txt";
	file.open(full_name, ios_base::out);
	
	
	for(Neuron &h : hiddenLayers[0])
	{
		for(i=0;i<h.weights.size();i++)
		{
			file << h.weights[i] << "\n";
		}
	}
	
	for(j=1;j<hiddenLayers.size();j++)
	{
        for(Neuron &h : hiddenLayers[j])
		{
			for(i=0;i<h.weights.size();i++)
		    {
			    file << h.weights[i] << "\n";
		    }
		}
	}
	
	for(Neuron &o : out_neurons)
	{
		for(i=0;i<o.weights.size();i++)
		{
			file << o.weights[i] << "\n";
		}
	}   
	file.close();	
}//End of func


void Net::Train(int iter, int speed)
{
	int i;
	for(i=0;i<iter;i++)
	{
		Forward_Prop();
		Back_Prop(speed);
	}
}

vector<double> Net::Run()
{
	Forward_Prop();
	return out_outputs;
}
