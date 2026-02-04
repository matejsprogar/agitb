#include "include/agitb.h"

class MyAGI
{
    using MyInput = std::bitset<10>;

public:
    bool operator==(const MyAGI& rhs) const { 
        // TODO
        return true; 
    }

    MyInput operator()(const MyInput& p) {
        // TODO AGI magic here!
        return MyInput();
    }
};

int main()
{
    using AGITB = sprogar::AGI::TestBed<MyAGI>;

    AGITB::run();
    return 0;
}