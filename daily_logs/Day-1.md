**Day-1:**

**Agenda:**
To Perform analysis of the given data.

**What was done:**
The raw data was converted into a csv file by calculating the rolling RMS and Kurtosis values. The header is as follows:

file_index,timestamp_min,filename,ch1_rms,ch1_kurtosis,ch2_rms,ch2_kurtosis,ch3_rms,ch3_kurtosis,ch4_rms,ch4_kurtosis,ch5_rms,ch5_kurtosis,ch6_rms,ch6_kurtosis,ch7_rms,ch7_kurtosis,ch8_rms,ch8_kurtosis

Each bearing has 2 accelerometer values.

**Report:**
RMS vs time was plotted for all eight bearing channels.
The RMS values remain relatively stable across most of the experiment, with large spikes appearing only near the end, which matches late-stage bearing degradation.
No strong long-term growth trend is visible in RMS.

Kurtosis vs time was also plotted.
Unlike RMS kurtosis shows early impulsive spikes well before the final failure, indicating the presence of localized impacts and early fault signatures. These spikes grow in frequency and amplitude as the bearing approaches failure.

Based on this behavior:
CNN can be used as a baseline model, but it will only capture local patterns in short windows.
True degradation is temporal, evolving over thousands of windows.
Because of this, LSTM/GRU is a better architectural match, since it can learn trends across time instead of treating each window independently.

For now, the plan is:
Build a CNN baseline and apply pruning.
Then transition to LSTM/GRU, expecting significant improvements in early detection.

Graphs are stored in the EDA_analysis/ folder.

**Conclusion:**

The EDA clearly shows that impulsive behavior (fault precursors) appears much earlier than RMS changes. This confirms that temporal models will be necessary to detect faults before severe degradation occurs.

Maybe freqency domain analysis would yield the reasons for this but as the aim of this analysis is different , the analysis is concluded here for this reason.

