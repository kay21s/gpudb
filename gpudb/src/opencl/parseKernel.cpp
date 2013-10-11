/*
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;

const char * createProgram(string path, int * num){

	*num = 1;

	ifstream kernelFile(path.c_str(),ios::in);

	ostringstream oss;

	oss << kernelFile.rdbuf();
	string srcStdStr = oss.str();

	char * res = (char *)malloc(srcStdStr.length()+1);
	memset(res,0,srcStdStr.length()+1);

	strcpy(res,srcStdStr.c_str());

	return res;
}

const char * createProgramBinary(string path, size_t * size){
	FILE *fp = fopen(path.c_str(),"r");

	char * res ;

	if(fp == NULL){
		printf("Failed to open binary kernel file\n");
		exit(-1);
	}

	fseek(fp,0,SEEK_END);
	*size = ftell(fp);
	rewind(fp);

	res = (char *) malloc(*size);

	if(!res){
		printf("Failed to allocate space for binary code\n");
		exit(-1);
	}

	fread(res,1,*size,fp);
	fclose(fp);

	return res;
	
}
