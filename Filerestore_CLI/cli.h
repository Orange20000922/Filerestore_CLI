#pragma once
#include <string>
#include <queue>
#include <vector>
#include <Windows.h>
#include "ImageTable.h"
using namespace std;
class CLI
{
private:
    static  vector<queue<string>> commands ;
	static  queue<string> args;
	static  bool initialized;  // 防止重复注册
	ImageTableAnalyzer* analyzer = new ImageTableAnalyzer();
	vector<LPVOID> argsinstances = vector<LPVOID>();
	public:
		void Run(string& command);
		static queue<string> SplitString(string& str, char delimiter);
	    static void ParseCommands(string& thecommmand,LPVOID instanceptr);
		static vector<queue<string>> GetCommands() { return commands; };
public:
	CLI();
	~CLI();
};