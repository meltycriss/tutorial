#!/usr/bin/perl

if (open(MYFILE, ">tmp")){
  $~ = "MYFORMAT";
  write MYFILE;
  format MYFILE =
  ==================================
            Hello, MYFILE!
  ==================================
.
  close MYFILE;
}
