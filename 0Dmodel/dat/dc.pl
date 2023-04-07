#!/usr/bin/perl

# a file to pick out selected species for plotting from UMIST data files

# Data files output from chemical model:
open FILE1, 'dc.out' or die $!;
#open FILE2, 'csfrac_s_orich_1.2_20jul.out' or die $!;
#open FILE2, './results/isob/fo/at10d2e4.dat' or die $!;
#open FILE3, 'abund10_1e6slow_mod.dat' or die $!;
#open FILE4, 'abund10_1e6slow_mod2.dat' or die $!;

# Files containing species we want to plot:
open FILE7, '> dc_molecules.dat' or die $!;
#open FILE8, '> csfrac_s_orich_1.2_plot_20jul.dat' or die $!;
#open FILE8, '> ./results/isob/graint10d2e4.dat' or die $!;
#open FILE9, '> ab10_1e6slow_mod.dat' or die $!;
#open FILE10, '> ab10_1e6slow_mod2.dat' or die $!;

# Species we want to plot:
@dspec1=qw(C+ C CO C2H HC3N HC5N CH3OH e- );

# reading fractionation files:
$fg=1;
while(<FILE1>) {
    chop;
    s/ +/,/g;
    @tp=split(/,/,$_);
    if (/TIME/) {
# here array 's' is only undefined if we're in the first block
	if ($s[1] eq "TIME") { $fg=0; }
	@s=@tp;
    } elsif (/\./) {
	for $i ( 2 .. 11 ) {
# 'abd1' is a hash of arrays of species abundances. @s is the array of species names for the current data block. the current line of the current data block is @tp. this line pushes the current line of numbers onto the arrays held in %abd1.
	    push(@{ $abd1{$s[$i]} },$tp[$i]);
	}
# in the first block get timesteps
	if ($fg) {
	    push(@time1,$tp[1]);
	}
    }
}
@test=keys(%abd1);
@test2=@{ $abd1{"H2D+"} };
print "no of timesteps in frac files is $#time1 +1 ($#test2 +1 for H2D+), no of specs is $#test +1\n";
print "\n";
$fg=1;
while(<FILE2>) {
    chop;
    s/ +/,/g;
    @tp=split(/,/,$_);
    if (/TIME/) {
# here array 's2' is only undefined if we're in the first block
	if ($s2[1] eq "TIME") { $fg=0; }
	@s2=@tp;
    } elsif (/\./) {
	for $i ( 2 .. 11 ) {
	    push(@{ $abd2{$s2[$i]} },$tp[$i]);
	}
# in the first block get timesteps
	if ($fg) {
	    push(@time2,$tp[1]);
	}
    }
}
$fg=1;
while(<FILE3>) {
    chop;
    s/ +/,/g;
    @tp=split(/,/,$_);
    if (/RADIUS/) {
# here array 's3' is only undefined if we're in the first block
	if ($s3[1] eq "RADIUS") { $fg=0; }
	@s3=@tp;
    } elsif (/\./) {
	for $i ( 2 .. 11 ) {
	    push(@{ $abd3{$s3[$i]} },$tp[$i]);
	}
# in the first block get timesteps
	if ($fg) {
	    push(@time3,$tp[1]);
	}
    }
}
$fg=1;
while(<FILE4>) {
    chop;
    s/ +/,/g;
    @tp=split(/,/,$_);
    if (/TIME/) {
# here array 's4' is only undefined if we're in the first block
	if ($s4[1] eq "TIME") { $fg=0; }
	@s4=@tp;
    } elsif (/\./) {
	for $i ( 2 .. 11 ) {
	    push(@{ $abd4{$s4[$i]} },$tp[$i]);
	}
# in the first block get timesteps
	if ($fg) {
	    push(@time4,$tp[1]);
	}
    }
}

@snames=keys(%abd1);
print "timesteps in time4 $#time4+1\n";

$j=0;
foreach $t (@time1) {
#    $t1 = $t + 5.01e+4;
    print FILE7 "$t ";
    foreach $d (@dspec1) {
	if (exists $abd1{$d}) {
	    @s=@{ $abd1{$d} };
	    print FILE7 "$s[$j] ";
	}
    }
    print FILE7 "\n";
    $j=$j+1;
}
$j=0;
foreach $t (@time2) {
    print FILE8 "$t ";
    foreach $d (@dspec1) {
	@s=@{ $abd2{$d} };
	print FILE8 "$s[$j] ";
    }
    print FILE8 "\n";
    $j=$j+1;
}

$j=0;
foreach $t (@time3) {
    print FILE9 "$t ";
    foreach $d (@dspec1) {
	@s=@{ $abd3{$d} };
	print FILE9 "$s[$j] ";
    }
    print FILE9 "\n";
    $j=$j+1;
}
$j=0;
foreach $t (@time4) {
    print FILE10 "$t ";
    foreach $d (@dspec1) {
	@s=@{ $abd4{$d} };
	print FILE10 "$s[$j] ";
    }
    print FILE10 "\n";
    $j=$j+1;
}
