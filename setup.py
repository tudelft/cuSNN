import os
import sys

cmake_add = ''
install_add = 'sudo '
if len(sys.argv) > 1: 
	install_add = ''
	cmake_add = '-DCMAKE_INSTALL_PREFIX=' + sys.argv[1] + ' '
os.system('rm -rf build/')
os.system('mkdir build/')
os.chdir('build/')
os.system('cmake ' + cmake_add + '../')
os.system(install_add + 'make install')
if len(sys.argv) <= 1:
	os.system(install_add + '/sbin/ldconfig')