#!/usr/bin/perl

$str = "abc_efg_criss_kkk";

$str =~ /[a-zA-Z]{5}(?=_)/;

#print "$`\n"
print "$&\n";
#print "$'\n";
print "$str\n";
