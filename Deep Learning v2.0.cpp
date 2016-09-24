#include <iostream>
#include <math.h>
#include <vector>
#include <stdio.h>      
#include <stdlib.h>     
#include <time.h> 

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
}
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
}
void Net::Back_Prop(double speed)
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

	j=0;
/**	for(Neuron &o : out_neurons)
	{
		cout << "Error of " << j+1 << ":"<< o.Cost(outputs[j]) << endl;
		j++;
	}**/
}



int main()
{
	srand (time(NULL));
	vector<double> inputs;
	vector<double> outputs;
	int j;
	inputs.push_back(1);//Give inputs and desired outputs
	inputs.push_back(0.5);
	inputs.push_back(0.1);
	inputs.push_back(0.2);
	
	outputs.push_back(0.6);



	
	Net Net1(inputs,1,3,outputs);//Define net
	for(j=0;j<100000;j++)
	{
	   Net1.Forward_Prop();
	   Net1.Back_Prop(2);
	   for(double &i : Net1.out_outputs)
	   {
	   	 cout << i << endl;
	   }
	}
	
	
	return 0;
}
