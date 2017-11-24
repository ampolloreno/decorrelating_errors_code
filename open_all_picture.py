import os
from subprocess import Popen
for thing in os.listdir():
	if os.path.isdir(thing) and 'report' in thing:
		Popen(['eog', thing + '/control_dpn_all.png'])
