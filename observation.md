# Benchark Observations

## Sequential CPU vs Parallel CPU

Benchmark:

```txt
Test Case                Implementation      Time (ms)      Passed
------------------------------------------------------------------
5x5 Edge Detect          Sequential CPU      0.017          Yes
5x5 Edge Detect          OpenMP CPU          0.005          Yes
5x5 Edge Detect          OpenCL GPU          0.015          Yes
1024x1024 Identity       Sequential CPU      183.146        Yes
1024x1024 Identity       OpenMP CPU          198.047        Yes
1024x1024 Identity       OpenCL GPU          0.231          Yes
2048x2048 Box Blur       Sequential CPU      757.920        Yes
2048x2048 Box Blur       OpenMP CPU          767.247        Yes
2048x2048 Box Blur       OpenCL GPU          1.393          Yes
```
As you can see, parallel sees performance degrade compared to sequential despite using 16x more threads, why? Isn't OpenMP supposed to make it much faster?

It points to one of the most important concepts in parallel programming: **overheads and bottlenecks**.

The short answer is: The code isn't *computation-bound*; it's **memory-bound**. OpenMP is making all 16 threads "fight" for the same resource (system RAM), and this data "traffic jam" is slower than just letting one core do the work.

Here’s a more detailed breakdown:

### 1. The Memory Bottleneck (The "Traffic Jam")

The convolution operation is extremely "data-hungry." For a 3x3 kernel, to calculate just **one** output pixel, the CPU must:
* **Read** up to 9 `input` values.
* **Read** 9 `kernel` values (though these are quickly cached).
* **Write** 1 `output` value.

This is a very high ratio of memory operations to calculations.

* **Sequential CPU:** One core is running. It can use its "private" L1 and L2 caches effectively. When it needs data from main RAM, it has relatively uncontested access to the memory bus (the "highway" to RAM).
* **OpenMP CPU (16 Threads):** You've launched 16 threads, likely on 16 different cores. All 16 cores are now *simultaneously* trying to read massive chunks of the `input` array and write to the `output` array. They are all competing for the **same shared memory bus**.
* This competition creates a massive bottleneck. The cores spend most of their time *stalling*—waiting for their turn to get data from RAM. The overhead of managing the threads *plus* this massive memory stall makes the parallel version significantly *slower* than the sequential one.

### 2. Why the Small 5x5 Test *Was* Faster

This is the key piece of evidence! For the `5x5 Edge Detect` test, the **entire dataset** (5x5 input, 5x5 output, 3x3 kernel) is incredibly small. It fits completely inside the ultra-fast **L1 or L2 cache** of the CPU.

In this case:
1.  The threads are created.
2.  The data is loaded into the caches (which is very fast).
3.  Because the data is *already in the cache*, there is no "traffic jam" on the memory bus.
4.  The 5 rows of work are split among the threads and executed in parallel *without* waiting for RAM.
5.  Here, the parallel computation speedup *does* outweigh the small overhead of creating the threads.

### 3. Why the GPU (OpenCL) is So Fast

This benchmark also perfectly illustrates *why* GPUs exist. A GPU is fundamentally different:
* **Massive Memory Bandwidth:** A graphics card has its own dedicated, extremely high-speed memory (VRAM) and a much, much "wider" memory bus. It's like having a 64-lane highway instead of a 4-lane one.
* **Latency Hiding:** It's *designed* for memory-bound tasks. It runs thousands of threads, and when one group of threads is stalled (waiting for memory), the GPU's scheduler simply switches to another group that's ready to compute.

For the large 1024x1024 test, the OpenCL version:
1.  Pays a one-time cost to copy the data to the GPU's VRAM (this is part of its overhead).
2.  Launches thousands of threads that run on specialized cores, which are fed data by the high-bandwidth VRAM.
3.  Crushes the memory-bound problem because its hardware is built for it.
4.  Pays a small cost to copy the result back.

**In summary:** OpenMP isn't a magic "go faster" button. It only speeds things up if the **CPU cores** are the bottleneck. In this case, the **memory bus** was the bottleneck, and adding more threads just made the bottleneck worse.