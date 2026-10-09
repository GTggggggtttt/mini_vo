//测试多线程

#include<iostream>
#include<thread>
#include<mutex>
#include<condition_variable>

//1.使用函数指针
// void printMessage(int count)
// {
//     for(int i=0;i<count;++i)
//     {
//         std::cout<<"Hello from thread(function pointer)!\n";
//     }
// }

// int main()
// {
//     std::thread t1(printMessage, 5);
//     t1.join();
//     return 0;
// }

//2.使用函数对象
// class PrintTask
// {
// public:
//     void operator()(int count)
//     const{
//         for(int i=0;i<count;++i)
//         {
//             std::cout<<"Hello from thread(function object)!\n";
//         }
//     }
// };

// int main()
// {
//     std::thread t2(PrintTask(),5);
//     t2.join();
// }

//3.使用Lambda表达式
// int main()
// {
//     std::thread t3
// }


// void increment(int &x)
// {
//     ++x;
     
// }

// int main()
// {   
//     int num=0;
//     std::thread t(increment,std::ref(num));
//     t.join();
//     std::cout<<"Value after increment: "<<num<<std::endl;
//     return 0;
// }


//4.线程同步与互斥
// std::mutex mtx;//全局互斥量

// void safeFunction(){
//     mtx.lock();//请求锁定全局互斥量
    
//     //访问或修改共享资源
//     mtx.unlock();//释放互斥量
// }


// int main()
// {
//     std::thread t1(safeFunction);
//     std::thread t2(safeFunction);
//     t1.join();
//     t2.join();
//     return 0;
// }



//4.3条件变量
// std::mutex mtx;
// std::condition_variable cv;//条件变量 全局的
// bool ready =false;

// void workerThread(){
//     std::unique_lock<std::mutex> lk(mtx);
//     cv.wait(lk,[]{return ready;});//等待条件
//     //当条件满足时执行工作
// }

// void mainThread(){
//     {std::lock_guard<std::mutex>lk(mtx);
//     //准备数据
//     ready = true;
//     }
//     cv.notify_one();//通知一个等待的线程
// }//离开作用域时的解锁


// using namespace std;
// void thread_1()
// {
//     cout<<"子线程1"<<endl;
// }

// void thread_2(int x)
// {
//     cout<<"x: "<<x<<endl;
//     cout<<"子线程2"<<endl;
// }

// int main()
// {
//     thread first(thread_1);//开启第一个线程，调用：thread_1()
//     thread second(thread_2,100);//开启线程，调用：thread_2(100)
//     // thread third(thread_2,3);
//     cout<<"主线程\n";

//     first.join();
//     second.join();
//     cout<<"子线程结束.\n";//必须等join结束
//     return 0;
// }

using namespace std;
void thread_1()
{
  while(1)
  {
  //cout<<"子线程1111"<<endl;
  }
}
void thread_2(int x)
{
  while(1)
  {
  //cout<<"子线程2222"<<endl;
  }
}
int main()
{
    thread first ( thread_1); // 开启线程，调用：thread_1()
    thread second (thread_2,100); // 开启线程，调用：thread_2(100)

    first.join(); // pauses until first finishes 这个操作完了之后才能destroyed
    second.join(); // pauses until second finishes//join完了之后，才能往下执行。
    while(1)
    {
      std::cout << "主线程\n";
    }
    return 0;
}