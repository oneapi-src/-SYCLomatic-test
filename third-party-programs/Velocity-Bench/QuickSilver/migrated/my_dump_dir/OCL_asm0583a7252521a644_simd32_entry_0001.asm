//.kernel _ZTSZ13warmup_kernelvEUlN4sycl3_V17nd_itemILi3EEEE_
//.platform DG2
//.thread_config numGRF=128, numAcc=4, numSWSB=16
//.options_string "-emitCrossThreadOffR0Reloc "
//.full_options "-emitLocation -enableCoalesceScalarMoves -enablePreemption -hasRNEandDenorm -noStitchExternFunc -useInlineData -emitCrossThreadOffR0Reloc -abortonspill -boundsChecking -presched-ctrl 22 -presched-rp 100 -nodpsendreorder -SBIDDepLoc -output -binary -dumpcommonisa -dumpvisa -printHexFloatInAsm -noverifyCISA -enableHalfLSC -waLscUgmFence -LSCFenceWA -insertRSDummyMov "
//.instCount 23
//.RA type	TRIVIAL_RA

//.declare BuiltInR0 (0)  rf=r size=32 type=ud align=16 words (r0.0)
//.declare  (1)  rf=r size=32 type=ud align=16 words (r2.0)
//.declare BuiltinA0 (2)  rf=a size=4 type=ud align=1 words (a0.0)
//.declare BuiltinA0Dot2 (3)  rf=a size=4 type=ud align=1 words (a0.2)
//.declare %null (9)  rf=r size=4 type=ud align=2 words
//.declare %local_id_x (12)  rf=r size=4 type=ud align=2 words (r1.2)
//.declare %local_id_y (13)  rf=r size=4 type=ud align=2 words (r1.3)
//.declare %local_size_x (14)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %local_size_y (15)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %group_id_x (16)  rf=r size=4 type=ud align=2 words (r0.1)
//.declare %group_id_y (17)  rf=r size=4 type=ud align=2 words (r0.6)
//.declare %group_id_z (18)  rf=r size=4 type=ud align=2 words (r0.7)
//.declare %group_count_x (19)  rf=r size=4 type=ud align=2 words (r1.0)
//.declare %group_count_y (20)  rf=r size=4 type=ud align=2 words (r1.1)
//.declare %tsc (21)  rf=r size=20 type=ud align=2 words
//.declare %arg (22)  rf=r size=0 type=ud align=16 words (r26.0)
//.declare %retval (23)  rf=r size=0 type=ud align=16 words (r26.0) Output
//.declare %sp (24)  rf=r size=8 type=uq align=4 words (r125.3)
//.declare %fp (25)  rf=r size=8 type=uq align=4 words (r125.2)
//.declare %sr0 (26)  rf=r size=16 type=ud align=2 words
//.declare %cr0 (27)  rf=r size=12 type=ud align=2 words
//.declare %ce0 (28)  rf=r size=4 type=ud align=2 words
//.declare %dbg0 (29)  rf=r size=8 type=ud align=2 words
//.declare implBufPtr (31)  rf=r size=8 type=uq align=4 words (r126.0)
//.declare localIdBufPtr (32)  rf=r size=8 type=uq align=4 words (r126.3)
//.declare %msg0 (33)  rf=r size=12 type=ud align=2 words
//.declare V0033 (41)  rf=r size=32 type=d alias=+0 align=16 words (r2.0)
//.declare V0035 (43)  rf=r size=32 type=d alias=+0 align=16 words (r2.0)
//.declare V0036 (44)  rf=r size=32 type=d align=16 words (r7.0)
//.declare V0037 (45)  rf=r size=12 type=d align=2 words (r8.0)
//.declare V0038 (46)  rf=r size=12 type=d align=2 words (r8.3)
//.declare V0039 (47)  rf=r size=32 type=w align=16 words (r1.0)
//.declare V0040 (48)  rf=r size=32 type=w align=16 words (r2.0)
//.declare V0041 (49)  rf=r size=32 type=w align=16 words (r3.0)
//.declare V0042 (50)  rf=r size=32 type=w align=16 words (r4.0)
//.declare V0043 (51)  rf=r size=32 type=w align=16 words (r5.0)
//.declare V0044 (52)  rf=r size=32 type=w align=16 words (r6.0)
//.declare  (53)  rf=r size=32 type=ud align=16 words (r127.0)
//.declare  (54)  rf=r size=32 type=ud align=16 words (r0.0)
//.declare  (55)  rf=r size=32 type=ud align=16 words (r127.0)
//.declare  (56)  rf=r size=32 type=ud align=16 words (r1.0)
//.declare  (57)  rf=r size=32 type=ud align=16 words (r7.0)
//.declare  (58)  rf=r size=128 type=ud align=16 words (r1.0)
//.declare  (59)  rf=r size=64 type=ud align=16 words (r5.0)
//.declare  (60)  rf=r size=32 type=ud align=16 words (r8.0)

// .inputs
// +----------+----------+--------+----------+------------+
// | id       | type     |  bytes | at       | class      |
// +----------+----------+--------+----------+------------+
// | V0039    | :w x 16  |     32 | r1       | general    |
// | V0040    | :w x 16  |     32 | r2       | general    |
// | V0041    | :w x 16  |     32 | r3       | general    |
// | V0042    | :w x 16  |     32 | r4       | general    |
// | V0043    | :w x 16  |     32 | r5       | general    |
// | V0044    | :w x 16  |     32 | r6       | general    |
// | V0036    | :d x 8   |     32 | r7       | general    |
// | V0037    | :d x 3   |     12 | r8       | general    |
// | V0038    | :d x 3   |     12 | r8+12    | general    |
// +----------+----------+--------+----------+------------+


// B000: Preds:{},  Succs:{B001}
per_thread_prolog:
(W)     mov (8|M0)               r127.0<1>:ud  0x0:ud                                                // 
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       // 
(W)     and (1|M0)               r127.0<1>:uw  r0.4<0;1,0>:uw    0xFF:uw                             // 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x20:uw              {I@2}          // 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x0:ud              {I@1}           // 
(W)     mad (1|M0)               r127.2<1>:ud  r127.2<0;0>:ud    r127.0<0;0>:uw    0xC0:uw              {I@1} // 
(W)     mov (8|M0)               r7.0<1>:ud    r1.0<1;1,0>:ud                                        // 
(W)     send.dc0 (8|M0)          r1       r127    null:0  0x0            0x024844FD           {A@1,$0} // wr:1h+0, rd:4; oword aligned block read x8 // 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x80:uw              {$0.src}       // 
(W)     send.dc0 (8|M0)          r5       r127    null:0  0x0            0x022843FD           {A@1,$1} // wr:1h+0, rd:2; oword aligned block read x4 // 
        nop                                                                                          // 
        nop                                                                                          // 
// B001: Preds:{B000},  Succs:{B002}
// cross_thread_prolog:
(W)     mov (8|M0)               r127.0<1>:ud  0x0:ud                              {$1.src}          // 
(W)     and (1|M0)               r127.2<1>:ud  r0.0<0;1,0>:ud    0xFFFFFFC0:ud                       // 
(W)     add (1|M0)               r127.2<1>:ud  r127.2<0;1,0>:ud  0x0:ud              {I@1}           // 
(W)     send.dc0 (8|M0)          r8       r127    null:0  0x0            0x021842FD           {A@1,$2} // wr:1h+0, rd:1; oword aligned block read x2 // 
// B002: Preds:{B001},  Succs:{}
// _main:
(W)     mov (8|M0)               r2.0<1>:ud    r0.0<1;1,0>:ud                   {$0.dst}             // $1
(W)     or (1|M0)                cr0.0<1>:ud   cr0.0<0;1,0>:ud   0x4C0:uw              {A@1}         // $1
        sync.nop                             null                             {Compacted,A@1}        // $2
        sync.nop                             null                             {Compacted,$2.src}     // $2
(W)     mov (8|M0)               r127.0<1>:f   r2.0<1;1,0>:f                    {Compacted,A@1}      // $2
(W)     send.gtwy (8|M0)         null     r127    null:0  0x0            0x02000010           {EOT,A@1} // wr:1+0, rd:0; end of thread // $2
L328:
        nop                                                                                          // $2


//.BankConflicts: 0
//.RMWs: 0
//


//.numALUInst: 17
//.numALUOnlyDst: 0
//.numALUOnlySrc: 1
//.accSubDef: 0
//.accSubUse: 0
//.accSubCandidateDef: 0
//.accSubCandidateUse: 0
//
//
//.singlePipeAtOneDistNum: 5
//.allAtOneDistNum: 4
//.syncInstCount: 2
//.tokenReuseCount: 0
//.AfterWriteTokenDepCount: 1
//.AfterReadTokenDepCount: 3
