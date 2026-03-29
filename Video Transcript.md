0:00
Hi and welcome back. Today's strategy is
0:02
something I'm trying for the very first
0:04
time. One of our viewers actually
0:06
encouraged me to test this branch of
0:07
artificial intelligence named genetic
0:10
programming and apparently we can use it
0:12
also for trading and the viewer even
0:14
shared some academic papers on the
0:16
subject and most of them report pretty
0:18
promising returns. So first of all,
0:20
thank you for sharing this very
0:21
informative idea. In this video, I'll
0:23
explain what genetic programming is, how
0:25
it works, and how we can actually code
0:27
it in Python. I'll show you which
0:29
libraries to use, how to train the
0:31
model, and of course, I'll share the
0:33
full back testing code and the trained
0:35
model with you. Quick heads up though,
0:37
training this model took me several days
0:39
on a Core i9 processor. So yes, it's
0:41
computationally heavy and we will see in
0:43
a moment why that is. Just keep in mind
0:45
that it's only the training phase. The
0:47
inference later on using the model is
0:50
much faster. So what is genetic
0:52
programming? Think of it as a branch of
0:54
AI that borrows ideas from evolution and
0:56
genetics. So instead of starting with a
0:58
fixed indicator, we let the computer
1:00
randomly generate thousands of formulas.
1:03
Each formula is a potential trading
1:05
indicator. These formulas and indicators
1:07
can evolve through generations. And each
1:10
indicator is back tested on historical
1:12
data to check its performance. So the
1:15
first step is to generate a pool of
1:16
random indicators that we will call the
1:18
initial population. This can be 5,000,
1:21
10,000 indicators or even more. In this
1:23
example, I took 15,000 as a start. each
1:26
formula or indicator is of different
1:28
complexity. Then we will back test each
1:30
of these indicators within the same
1:32
trading strategy on historical data. And
1:35
this is why it's computationally
1:37
expensive because in this example with
1:40
these numbers that I'm presenting here,
1:42
we are applying for one iteration 15,000
1:45
back tests. So imagine we are running
1:47
this for around 20 generations. That's
1:50
20 * 15,000 back tests in total. In the
1:53
second step, we identify the most
1:55
promising indicators through a selection
1:58
process based on the back test's
1:59
performance. We will keep the top 10% or
2:03
any percentage actually. This is just a
2:05
parameter that can be changed in the
2:06
simulation. And we simply discard the
2:09
remaining 90% of these formulas or
2:11
indicators. And all of this selection is
2:13
of course based on either the sharp
2:15
ratio or the returns percentage or the
2:17
risk adjusted returns. Then in the third
2:19
step, we breed the uh selected
2:22
indicators together. In other words, we
2:23
randomly mix parts of one formula with
2:26
parts of another just like DNA crossover
2:28
and we add small mutations or random
2:31
changes. To recap the process, we have
2:34
our first generation of indicators. We
2:36
back test these and we select the top
2:38
10%. Then we start shuffling parts of
2:40
these selected indicators to generate a
2:43
new set of 15,000 indicators. That is
2:45
the second generation of our mutated
2:47
formulas that are the descendants of the
2:50
top 10% selection. So these are supposed
2:53
to be better set of indicators or the
2:55
second generation is supposed to be
2:57
better than the first generation in
2:59
terms of returns or sharp ratio in the
3:01
back testing. Then we repeat the back
3:03
test with the new indicators. We select
3:05
10% again. We continue this process for
3:08
few generations and after 10 20
3:11
generations for example indicators
3:13
evolve through these steps and so we
3:15
expect them to be more performant then
3:17
we end up choosing the champion
3:18
indicator from the last generation. It's
3:21
kind of looking for this one best
3:23
formula that could survive the entire
3:25
process. And once we have this champion
3:27
we test it on completely new unseen data
3:30
to see if it holds up. We'll go through
3:32
the details later on in the code. Now
3:34
here's the twist. These aren't your
3:36
usual technical indicators. They are not
3:39
RSI nor MACD nor binger bands. These are
3:42
brand new machine generated indicators.
3:45
They are randomly generated every time
3:47
we run the code. Some papers use this
3:49
genetic approach. Taking it a step
3:51
further. Instead of using one asset,
3:54
they feed in prices from multiple
3:56
assets. For example, they might give the
3:58
model US dollar, Japanese yen, a British
4:00
pound, US dollar data, and so on. in
4:03
order to evolve an indicator to predict
4:05
the euro US dollar prices based on
4:08
prices of different assets. And that's
4:10
only one approach. Another idea is to
4:12
use classic technical indicators as
4:14
inputs and let genetic programming
4:16
shuffle and mutate them into a brand new
4:19
formula that is supposed to be more
4:21
performant. As you can see, it can get
4:23
pretty wild very quickly. I'm not going
4:25
to cover every single variation
4:26
mentioned in the papers right now.
4:29
Instead, we will start with one version,
4:32
get some results, and then maybe revisit
4:34
this in future videos based on your
4:35
feedback. I personally find this
4:37
fascinating. It's fun. It's
4:39
experimental, and it opens the door to
4:41
totally new trading ideas. Again, the
4:44
Python code and the trained model as
4:46
well as the environment requirements are
4:48
shared on my GitHub repo. The link is
4:50
available in the description of this
4:52
video. So feel free to try it out. Play
4:54
the parameters, brainstorm new ideas,
4:56
and share your experiments in the
4:58
comments. All right, let's jump in and
5:00
I'll show you how we coded this whole
5:02
thing in Python. Then we can back test
5:04
the model and check the results. So this
5:06
is the uh Python script. It loads the 5m
5:09
minute time frame candlesticks. These
5:13
are CSV files of four pairs Euro US
5:15
dollar. Uh GBP US dollar, Australian
5:19
dollar and US dollar and Japanese yen as
5:22
well. So I have one, two, three and four
5:25
files. These are my data files. It
5:28
standardizes the open, high, low, close
5:31
um column names. Merges all pairs into
5:34
one data frame. Then splits the data set
5:36
into train uh validation and test. These
5:39
are the dates where the data is being
5:42
split. We will evolve on train then
5:44
choose the winner on validation the
5:46
winner indicator and then report real
5:49
performance on the test or out of sample
5:52
data. And here we can see the uh
5:55
population size and the number of
5:56
generations. So for this example I took
5:58
1,00 uh population 1,000 indicators at
6:02
first and then 15 generations for uh
6:05
evolution. The crossing uh percentage is
6:08
90% and the mutation percentage is 15%.
6:12
Then we have the maximum depth of each
6:15
indicator. So these are basically the
6:17
number of mathematical operations that
6:19
we can apply on the input parameters and
6:23
the maximum length of each indicator. We
6:25
have the initial cash cache for the back
6:27
testing, the commission percentage, the
6:30
no trade band. I'm going to explain this
6:32
in a while and the uh position uh grid.
6:35
So basically each indicator is a math
6:37
expression tree built from primitives
6:39
like add, subtract and uh s cosine and
6:43
mathematical operations that you can see
6:45
here in this part. So we're going from
6:48
here down to um line 136 almost. Inputs
6:54
are the open, high, low, close values
6:56
from all pairs. So the um genetics
6:58
program can make signals across markets.
7:01
indicator output is a time series of
7:04
desired percentage exposure between
7:07
minus 100 and plus 100%. So um it's
7:12
clipped to minus 100 and plus 100. This
7:15
is the desired uh percentage exposure.
7:18
As we can see here in line 167, it's
7:22
then converted to a value when we divide
7:25
it by 100 between minus1 and one. This
7:28
number represents the desired position
7:31
size. Plus 100 means fully long or + one
7:35
means fully long. Minus 100 means fully
7:39
short. Zero means no position. So we
7:42
convert this into a portfolio weights
7:45
between minus1 and one. Now to avoid
7:47
overtrading we apply something called a
7:50
deadbend. The deadband means if the new
7:53
position signal meaning if this number
7:56
between minus 100 plus 100 or between
7:58
minus1 and one is within let's say 10%
8:01
around the zero plus or minus 10% the
8:04
system does nothing. We're not allowing
8:06
any trades. It has to cross this 10%
8:09
threshold in a positive or negative
8:11
direction to go long or short. So trades
8:14
are only triggered when the signal
8:16
changes by more than the deadband
8:18
threshold of 10%. And again this
8:20
strategy was inspired by the publication
8:24
or the work published by these three
8:26
authors that was sent to me by one of
8:28
the viewers on this channel and we are
8:30
trying to experiment using their method.
8:32
Now if you have noticed I have two files
8:34
GP strategy progress vectorbt and GP
8:38
strategy progress. This one uses back
8:40
testing.py Pi which is more discreet in
8:42
the back testing process and this one
8:45
uses vector BT which is vectorzed and it
8:47
runs much faster which is needed
8:49
actually to uh back test and do this
8:52
kind of heavily uh heavy computation um
8:55
experiment but my preferred method even
8:57
if it's slower is this one because it's
8:59
discrete so using back testing.py
9:02
because for each trade we know what we
9:04
are doing and we're not really bulk
9:06
executing in a vectorzed way. And here
9:08
you can see how vectorbt is used in this
9:11
file or back testing.py in the previous
9:14
file. It's the same. So we are using um
9:18
target percent size type. Uh we have a
9:21
commission percentage. We have an
9:23
initial cache and the frequency is 5
9:25
minutes. We're using the 5 minutes time
9:27
frame. Now for each back test using an
9:31
individual indicator generated by this
9:33
genetic program algorithm the fitness of
9:36
the indicator is measured which is the
9:38
exponential negative the returns. It's
9:41
seen here. So it's exponential minus
9:43
total returns meaning for higher returns
9:46
we get smaller fitness which is better
9:48
because the genetics algorithm is
9:50
minimizing the fitness and this again is
9:52
also based on the paper where we can
9:55
find the fitness here that's u
9:58
expression two fitness is equal to
10:01
exponential minus the return then we can
10:04
move on to the evolution run uh which is
10:07
defined in this function. So run
10:10
evolution. We create a population of
10:12
1,000 indicators or strategies and
10:15
evolve over 15 generations. Each
10:17
generation does evaluate all strategies,
10:20
all indicators. Keep the best ones in a
10:23
hall of fame somehow. Select parents
10:26
somehow a tournament selection. Apply
10:28
crossover with 90% chance. Apply
10:30
mutation with 15% chance. I know they
10:33
don't add up, but it normalizes at some
10:35
point. and it re-evaluate any modified
10:38
strategies. So we rerun the back test on
10:41
any mutated or crossed uh strategies or
10:44
indicators and while running the um
10:46
evolution we print some progress
10:48
parameters like the minimum fitness the
10:50
average fitness and the evaluation uh
10:52
the evaluation count. So at the end we
10:55
take the hall of fame top strategies and
10:57
indicators score them on the validation
11:00
period pick the best one then we run one
11:02
final test using the out of sample back
11:05
test data on the test set meaning which
11:08
allows us to print uh key stats like
11:11
return sharp ratio trades and win rate
11:13
and the final equity and finally we are
11:16
saving whatever model we just trained
11:20
into um a file so best_individual
11:24
D in this case which we can load later
11:27
on without rerunning the whole evolution
11:29
process just for inference and using for
11:31
back tests. Now just for the sake of
11:33
showing you an example how it runs. I'm
11:35
going to run a small evolution. So this
11:38
is 1,00 uh population 15. Let's try it
11:42
out. See how long it will take. Of
11:44
course you can run it for more than
11:46
that. You can start with 75,000. This is
11:49
what the paper is advising. This is what
11:51
they did. 75,000
11:53
[snorts] uh population size as a start
11:55
and for 15 generations. So now we are
11:58
shrinking this because then it would
11:59
take uh too much time. So Python uh GP
12:03
strategy this is the vector BT version
12:07
and we can run this.
12:10
So you can see it's loading the data now
12:13
running the evolution and it's going to
12:15
print the steps in which the program is
12:18
going through. I've also added a Jupyter
12:20
notebook file where we repeat all of
12:23
these steps in simple cells making it as
12:26
concise and as clear as possible
12:29
including and importing functions
12:31
whenever needed from the other files. So
12:34
as you can see we have the u train
12:36
validation and test. We have all the
12:39
parameters as well and some helper
12:41
functions helper utilities here. uh and
12:44
the back testing part repeats the same
12:47
uh work we uh we have explained in the
12:49
other scripts only using a Jupyter
12:52
notebook file if you want to see the uh
12:54
different steps when you are running
12:56
this program. Uh anyway, robustness
12:58
should be tested definitely this is not
13:00
a magic u method to have the magic
13:03
indicators. It's simply an experiment.
13:05
We are trying something new here. Now if
13:07
you have a model that is already saved,
13:11
we can also run the trading system load
13:13
and infer. This uh file just loads a
13:17
file loads the uh model trading model
13:19
indicator if you've already uh done your
13:22
genetics process and generations and so
13:24
on and it will be using it just for a
13:27
back test on the testing uh uh period or
13:29
the testing data. So let me run one test
13:32
for you just to show you how it's
13:34
working. So it's loading the data. It
13:36
gives you the uh fitness of uh the
13:39
indicator it's going to use which is the
13:41
best model basically. So it's not really
13:43
the best model. It's a fitness of 10 to
13:45
six. So this is a very bad indicator,
13:47
very bad uh model. Remember the fitness
13:50
should be something very low, but we're
13:52
going to run it anyway just for the sake
13:54
of the experiment. We're starting with a
13:56
million dollar in cache as a start. It's
14:00
loading from best individual.
14:02
Dill file, which is right here. I have
14:05
two other models individual one
14:07
individual two. So these are different
14:09
runs with different generations,
14:11
different settings which is uh very time
14:13
consuming when you are training but you
14:16
can save them for uh later use. So now
14:19
I'm loading this one and it's going to
14:21
come up with the equity the back test.
14:23
It's going to show the equity curve and
14:25
so the back test results here are 46% in
14:29
return sharp ratio 1.1 and maximum
14:31
drawdown of minus 12%. So this is kind
14:35
of a lucky indicator or strategy because
14:38
the fitness is supposed to be very high.
14:39
So it's not supposed to be a good
14:41
indicator. Somewhere in the back test it
14:43
was faulty but actually it's giving a
14:45
positive uh result on the testing uh
14:49
data. So it's saved here. I could use it
14:52
later on maybe on another back test see
14:54
if it works. But for now it's just a
14:56
simple example. Now the way you can
14:59
evolve this is change the strategy.
15:01
Remember that we are using uh some kind
15:03
of a percentage indicator. If it's plus
15:07
10% and above, we're going long, minus
15:10
10%, we're going short and so on. Things
15:13
can be changed. We can use a take-profit
15:15
stop-loss like go back to a normal
15:17
strategy and just use genetics
15:19
programming to train an indicator. But
15:21
that might be for a next video. For this
15:23
video, I think this will be it. I hope
15:25
it inspired your curiosity. I'll be
15:27
sharing the code files on a GitHub
15:29
repository so you can download them,
15:31
experiment with different parameters and
15:33
see how it works. Until our next one,
15:35
trade safe and see you next time.
