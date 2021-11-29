Bach10-mf0-synth
=============

Bach10-mf0-synth (c) by Justin Salamon, Rachel Bittner, Jordi Bonada, Juan Jose Bosch, Emilia Gómez and Juan Pablo Bello.
Bach10-mf0-synth is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). 
You should have received a copy of the license along with this work. If not, see http://creativecommons.org/licenses/by-nc/4.0/


Created By
----------

Justin Salamon*, Rachel Bittner*, Jordi Bonada^, Juan Jose Bosch^, Emilia Gómez^ and Juan Pablo Bello*.
* Music and Audio Research Lab (MARL), New York University, USA
^ Music Technology Group, Universitat Pompeu Fabra, Spain
http://synthdatasets.weebly.com/
http://steinhardt.nyu.edu/marl/
https://www.upf.edu/web/mtg

Version 1.0.0


Description
-----------

Bach10-mf0-synth contains 10 classical music pieces (four-part J.S. Bach chorales) from the Bach10 dataset 
(http://music.cs.northwestern.edu/data/Bach10.html) where each instrument (basoon, clarinet, saxophone and violin) has 
been resynthesized to obtain perfect f0 annotations using the analysis/synthesis method described in the following 
publication:

J. Salamon, R. M. Bittner, J. Bonada, J. J. Bosch, E. Gómez, and J. P. Bello. "An analysis/synthesis framework for 
automatic f0 annotation of multitrack datasets". In 18th Int. Soc. for Music Info. Retrieval Conf., Suzhou, China, 
Oct. 2017.

This dataset includes:
* 10 mono wav files of mixes where each instrument (basoon, clarinet, saxophone and violin) has been resynthesized 
  using the analysis/synthesis method described in the paper.
* 10 csv files containing a perfect multiple-f0 annotation of all the instruments in the mix, obtained via the 
  analysis/synthesis method described in the paper
* 40 mono wav files consisting of the individual solo stems (tracks) for every instrument in each of the 10 pieces, 
  resynthesized using the analysis/synthesis method described in the paper.
* 40 csv files containing a perfect f0 annotation for every instrument in each of the 10 pieces, obtained using the 
  analysis/synthesis method described in the paper.

The data come in four folders, the contents of which is described below.


audio_mix
---------
Contains 10 mono wav files of mixes where each instrument (basoon, clarinet, saxophone and violin) has been 
resynthesized using the analysis/synthesis method described in the paper. The mix is obtained by taking an unweighted
sum of the resynthesized solo stems (tracks).

Naming convention: 
<pieceID>_<title>_MIX_mf0synth.wav

Example: 
01_AchGottundHerr_MIX_mf0synth.wav


annotation_mf0
--------------
Contains 10 csv files containing a perfect multiple-f0 annotation of all instruments in the mix, obtained via the 
analysis/synthesis method described in the paper. 

Format:
The annotations follow the MIREX multiple-f0 estimation (frame-basis) format:
https://www.music-ir.org/mirex/wiki/2018:Multiple_Fundamental_Frequency_Estimation_%26_Tracking#I.2FO_format
This format is also support by mir_eval: https://github.com/craffel/mir_eval

Each row in the annotation starts with a timestamp, followed by 0 or more tab separated frequency values in Hz 
representing the f0 of each active pitched instrument present in the time frame represented by the row. The hop size 
of the annotation is exactly 10 ms. 

IMPORTANT: no assumptions can be made as to the ordering of the f0 values in each row. The frequency values are NOT 
ordered neither by instrument nor by frequency, and should thus be treated as a "bag of frequencies" (a set) without 
any assumptions as to which frequency belongs to which instrument.

Naming convention:
<pieceID>_<title>_MIX_mf0synth.csv

Example:
01_AchGottundHerr_MIX_mf0synth.csv


audio_stems
-----------
Contains 40 mono wav files consisting of the individual solo stems (tracks) for every instrument in each of the 
10 pieces, resynthesized using the analysis/synthesis method described in the paper.

Naming convention: 
<pieceID>_<title>_<instrument>.RESYN.wav

Example: 
01_AchGottundHerr_bassoon.RESYN.wav


annotation_stems
----------------
Contains 40 csv files containing a perfect f0 annotation for every instrument in each of the 10 pieces, obtained using 
the analysis/synthesis method described in the paper.

Format:
Each file contains two comma-separated columns, the first containing timestamps and the second containing the stem 
f0 in Hz. The first frame in the annotation is zero-centered. Silence is indicated as 0 Hz. The hop size of the 
annotation is 128/44100 seconds (~2.9 ms).

Naming convention:
<pieceID>_<title>_<instrument>.RESYN.csv

Example:
01_AchGottundHerr_bassoon.RESYN.csv


Please Acknowledge Bach10-mf0-synth in Academic Research
--------------------------------------------------------

Please cite the following publication when using Bach10-mf0-synth:

J. Salamon, R. M. Bittner, J. Bonada, J. J. Bosch, E. Gómez, and J. P. Bello. "An analysis/synthesis framework for 
automatic f0 annotation of multitrack datasets". In 18th Int. Soc. for Music Info. Retrieval Conf., Suzhou, China, 
Oct. 2017.

For information about the original Bach10 dataset please see (and cite):

Z. Duan, B. Pardo, and C. Zhang. "Multiple fundamental frequency estimation by modeling spectral peaks and non-peak 
regions". IEEE Trans. on Audio, Speech, and Language Processing, 18(8):2121–2133, 2010.


Conditions of Use
-----------------

Dataset created by Justin Salamon, Rachel Bittner, Jordi Bonada, Juan Jose Bosch, Emilia Gómez and Juan Pablo Bello. 
 
The Bach10-mf0-synth dataset is offered free of charge under the terms of the Creative Commons
Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0): http://creativecommons.org/licenses/by-nc/4.0/
 
The dataset and its contents are made available on an "as is" basis and without warranties of any kind, including 
without limitation satisfactory quality and conformity, merchantability, fitness for a particular purpose, accuracy or 
completeness, or absence of errors. Subject to any liability that may not be excluded or limited by law, NYU is not 
liable for, and expressly excludes, all liability for loss or damage however and whenever caused to anyone by any use of 
the Bach10-mf0-synth dataset or any part of it.


Feedback
--------

Please help us improve Bach10-mf0-synth by sending your feedback to: justin.salamon@gmail.com
In case of a problem report please include as many details as possible.
