#pragma once

#include <iostream>
#include <fstream>

//--- Loader for .xyz files

bool load_grid_xyz(std::string filename, float* &data, unsigned int& sizeX, unsigned int& sizeY, unsigned int& sizeZ) {
	double voxelSizeX = 0;
	double voxelSizeY = 0;
	double voxelSizeZ = 0;

	std::ifstream in(filename, std::ifstream::in | std::ifstream::binary);
	if (!in.good())
	{
		return false;
	}

	in.read(reinterpret_cast<char*>(&sizeX), sizeof(unsigned int));
	in.read(reinterpret_cast<char*>(&sizeY), sizeof(unsigned int));
	in.read(reinterpret_cast<char*>(&sizeZ), sizeof(unsigned int));
	in.read(reinterpret_cast<char*>(&voxelSizeX), sizeof(double));
	in.read(reinterpret_cast<char*>(&voxelSizeY), sizeof(double));
	in.read(reinterpret_cast<char*>(&voxelSizeZ), sizeof(double));

	data = new float[sizeX * sizeY * sizeZ];

	float maximum = 0;

	for (unsigned int x = 0; x < sizeX; x++)
	{
		for (unsigned int y = 0; y < sizeY; y++)
		{
			for (unsigned int z = 0; z < sizeZ; z++)
			{
				float value;
				in.read(reinterpret_cast<char*>(&value), sizeof(float));
				unsigned int idx = x + sizeX * y + sizeX * sizeY * z;
				data[idx] = value;
				maximum = std::max(value, maximum);
			}
		}
	}

	maximum = std::max(0.00001f, maximum);

	for (unsigned int x = 0; x < sizeX; x++)
	{
		for (unsigned int y = 0; y < sizeY; y++)
		{
			for (unsigned int z = 0; z < sizeZ; z++)
			{
				unsigned int idx = x + sizeX * y + sizeX * sizeY * z;
				data[idx] /= maximum;
			}
		}
	}

	in.close();

	return true;
}