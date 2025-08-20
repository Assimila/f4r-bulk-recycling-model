;
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
begin
;
;************************************************
; read in netCDF file
;************************************************
directory = "/Users/ellendyer/Library/Mobile Documents/com~apple~CloudDocs/1SHARED_WORK/Work/3_ESA_GRANT/MODEL/data/ncep_test/"
   vf1  =  addfile(directory+"uwnd.mon.mean.nc","r")    ; Load netCDF file
   vf2  =  addfile(directory+"vwnd.mon.mean.nc","r")
   rhf  =  addfile(directory+"rhum.mon.mean.nc","r")
   msh  =  addfile(directory+"shum.mon.mean.nc","r")
   psfc =  addfile(directory+"pres.mon.mean.nc","r")
   hg   =  addfile(directory+"pres.mon.mean.nc","r")
   rtn  =  addfile(directory+"prate.mon.mean.nc","r")
   r1   =  addfile(directory+"lhtfl.mon.mean.nc","r")
;**********************************************************************

;do z = 240,251
;z = 243
   pre  = rtn->prate(time|240,lat|:,lon|:)
   rh1  = short2flt(rhf->rhum(240,0:7,{-7.5:7.5},:))
   U1   = short2flt(vf1->uwnd(240,0:7,{-7.5:7.5},:))      ; transform scale and offset 1013 to 300 hPa only
   V1   = short2flt(vf2->vwnd(240,0:7,{-7.5:7.5},:))      ; wind in " m/s "
   sh1  = short2flt(msh->shum(240,0:7,{-7.5:7.5},:))      ; specific humidity in " g/Kg "
   ps1  = psfc->pres(240,{-7.5:7.5},:)                    ; surface presure in " hPa "
   hgp  = hg  ->pres(240,:,:)
   Ei   = r1  ->lhtfl(240,:,:)                                ; latent heat flux in " W/m2 "

;********************************************************
   rh  = lonFlip(rh1)                                           ; Change  [0...375.5] to [-180.0...180.0]
   U   = lonFlip(U1)
   V   = lonFlip(V1)
   sh  = lonFlip(sh1)
   ps  = lonFlip(ps1)

;****************************************************************
      plvl   = rh&level  ; retrieve pressure levels in hPa

      shu = (sh)*U
      shv = (sh)*V

  copy_VarMeta(rh,shu)
  copy_VarMeta(rh,shv)
  shu@units = " g/Kg m/s"
  shv@units = " g/Kg m/s"

       linlog = 1
       pbot   = 1013
       ptop   = 300
       pwu   = shu(time|:,lon|:,lat|:,level|0)  ; create a variable with metadata
       pwv   = shv(time|:,lon|:,lat|:,level|0)  ;

pwu = 1.02*1e-5*vibeta(plvl,shu(time|:,lon|:,lat|:,level|:),linlog,ps(time|:,lon|:,lat|:),pbot,ptop)
pwv = 1.02*1e-5*vibeta(plvl,shv(time|:,lon|:,lat|:,level|:),linlog,ps(time|:,lon|:,lat|:),pbot,ptop)

       pwu@long_name = "total column moisture flux per second"
       pwu@units     = "m2/s"

rayon       = 6371220.          ; rayon terre en m�tre

dx     = rayon*2.5*3.1415927/180
dy     = rayon*2.5*3.1415927/180
;--------------------------------------------------------------
;
;  Interpolate Evaporation to 2.5�x2.5�
;
;-----------------------------------------------------------------

   E2   = (1e-3*Ei(time|:,lat|:,lon|:))/2510400           ; hauteur d'eau (m) �vapor�e par seconde (s)...........[m/s]
    copy_VarMeta(Ei(time|:,lat|:,lon|:),E2)

E2@long_name = " "               ; Evaporation interpolated to 2.5�x2.5� grid
E2@_FillValue = -9.96921e+36

latE2 = E2&lat(::-1)
lonE2 = E2&lon
LAT   = hgp&lat(::-1)
LON   = hgp&lon

Evp = linint2  (lonE2,latE2,E2(time|:,lat|:,lon|:), True, LON,LAT, 0)
Pre = linint2  (lonE2,latE2,pre(time|:,lat|:,lon|:), True, LON,LAT, 0)

Evp!0 = "time"
Evp!1 = "lat"
Evp!2 = "lon"

Pre!0 = "time"
Pre!1 = "lat"
Pre!2 = "lon"

Evp&time = E2&time
Evp&lat  = hgp&lat(:)
Evp&lon  = hgp&lon

Pre&time = E2&time
Pre&lat  = hgp&lat(:)
Pre&lon  = hgp&lon

Ev = lonFlip(Evp)         ; Change longitude range from [0...375.5] to [-180.0...180.0]
pr = lonFlip(Pre)

E1 = Ev(time|:,lon|:,lat|:)
Pr = pr(time|:,lon|:,lat|:)
;--------------------------------------        ;  End Interpolation

E   = E1( time|:,{lon|10.:32.5},{lat|-7.5:7.5})
PRE = Pr( time|:,{lon|10.:32.5},{lat|-7.5:7.5})
Fx  = pwu(time|:,{lon|10.:32.5},{lat|-7.5:7.5})
Fy  = pwv(time|:,{lon|10.:32.5},{lat|-7.5:7.5})

;print("Fx = "+Fx  +"  Fy = "+Fy   +"   E = "+E)


h   = dx
k   = dy
;K1  = 10000
K1  = 500
R   = 0.2

ntime = dimsizes(Fx&time)
nlon  = dimsizes(Fx&lon)
nlat  = dimsizes(Fx&lat)
N     = nlon
M     = nlat

u = new((/ntime,nlon,nlat-1/),float)
v = new((/ntime,nlon-1,nlat/),float)
e = new((/ntime,nlon-1,nlat-1/),float)
p = new((/ntime,nlon-1,nlat-1/),float)

Ao  = new(1,float)
A   = new((/ntime,nlon+2,nlat+2/),float)
B   = new((/ntime,nlon+2,nlat+2/),float)

do t = 0,ntime-1
   do i = 0,nlon -1
      do j = 0,nlat-2

          u(t,i,j) = 0.5*( Fx(t,i,j) + Fx(t,i,j+1))    

      end do
   end do
end do

do t = 0,ntime-1
   do i = 0,nlon-2
      do j = 0,nlat-1

          v(t,i,j) = 0.5*( Fy(t,i,j) + Fy(t,i+1,j))   

      end do
   end do
end do

e = Fx(time|:,lon|0:nlon-2,lat|0:nlat-2)

do t = 0,ntime-1                 ;  Evaporation field
   do i = 0,nlon-2
      do j = 0,nlat-2

         e(t,i,j) = ( E(t,i,j) + E(t,i,j+1) + E(t,i+1,j) + E(t,i+1,j+1))/4

      end do
   end do
end do

p = Fx(time|:,lon|0:nlon-2,lat|0:nlat-2)

do t = 0,ntime-1                 ;  compute Precipitation
   do i = 0,nlon-2
      do j = 0,nlat-2

        p(t,i,j) = e(t,i,j) + ((1/h)*( u(t,i,j) - u(t,i+1,j)) + (1/k)*( v(t,i,j) - v(t,i,j+1)))

      end do
   end do
end do

miss = -999.9
print( p@_FillValue)
do t = 0,ntime-1                 ;  replace negative values, if any
   do i = 0,nlon-2
      do j = 0,nlat-2
          if ((p(t,i,j).lt.0).and.(p(t,i,j).ne.miss)) then
      p(t,i,j) = 0
         end if
      end do
   end do
end do

do t = 0,ntime-1                 ;  initialisation recycling values
   do i = 0,nlon+1
      do j = 0,nlat+1

        B(t,i,j) = 0

      end do
   end do
end do


K = 0                          ;  iteration
do while (K.lt.K1)

print("K = "+K)

do t = 0,ntime-1
   do i = 2,nlon-2
      do j = 2,nlat-2
                             ; Inner cells

         if ((u(t,i-1,j-1).gt.0) .and. (v(t,i-1,j-1).gt.0)) then
Ao       = 2*p(t,i-1,j-1)*h*k  + u(t,i,j-1)*k + v(t,i-1,j)*h
A1       = 2*h*k*e(t,i-1,j-1)  - u(t,i,j-1)*k*B(t,i+1,j) - v(t,i-1,j)*h*B(t,i,j+1) + u(t,i-1,j-1)*k*(B(t,i,j) + B(t,i-1,j)) + v(t,i-1,j-1)*h*(B(t,i,j) + B(t,i,j-1))
C        = A1/Ao
B(t,i,j) = B(t,i,j) + R*((A1/Ao) - B(t,i,j))

         end if

         if ((u(t,i-1,j-1).gt.0) .and. (v(t,i-1,j-1).lt.0)) then
Ao       = 2*p(t,i-1,j-1)*h*k  + u(t,i,j-1)*k - v(t,i-1,j-1)*h
A1       = 2*e(t,i-1,j-1)*h*k - u(t,i,j-1)*k*B(t,i+1,j) + v(t,i-1,j-1)*h*B(t,i,j-1) + u(t,i-1,j-1)*k*(B(t,i,j) + B(t,i-1,j)) - v(t,i-1,j)*h*(B(t,i,j) + B(t,i,j+1))
B(t,i,j) = B(t,i,j) + R*(A1/Ao - B(t,i,j))

         end if

         if ((u(t,i-1,j-1).lt.0) .and. (v(t,i-1,j-1).gt.0)) then
Ao       = 2*p(t,i-1,j-1)*h*k - u(t,i-1,j-1)*k + v(t,i-1,j)*h
A1       = 2*e(t,i-1,j-1)*h*k - v(t,i-1,j)*h*B(t,i,j+1) + u(t,i-1,j-1)*k*B(t,i-1,j) - u(t,i,j-1)*k*(B(t,i,j) + B(t,i+1,j)) + v(t,i-1,j-1)*h*(B(t,i,j-1) + B(t,i,j))
B(t,i,j) = B(t,i,j) + R*(A1/Ao - B(t,i,j))

         end if

         if  ((u(t,i-1,j-1).lt.0) .and. (v(t,i-1,j-1).lt.0)) then
Ao       = 2*p(t,i-1,j-1)*h*k - u(t,i-1,j-1)*k - v(t,i-1,j-1)*h
A1       = 2*e(t,i-1,j-1)*h*k + u(t,i-1,j-1)*k*B(t,i-1,j) + v(t,i-1,j-1)*h*B(t,i,j-1) - u(t,i,j-1)*k*(B(t,i,j) + B(t,i+1,j)) - v(t,i-1,j)*h*(B(t,i,j) + B(t,i,j+1))
B(t,i,j) = B(t,i,j) + R*(A1/Ao - B(t,i,j))
         end if

        end do
     end do
  end do
                   ;**********     boundary cells   ***************
                   ;***   boundary East
                   ;*     i = 1

do t = 0,ntime-1
    do j = 1,nlat-1


         if ((u(t,0,j-1).gt.0) .and. (v(t,0,j-1).gt.0)) then
Ao       = 2*p(t,0,j-1)*h*k  + u(t,1,j-1)*k + v(t,0,j)*h
A1       = 2*h*k*e(t,0,j-1)  - u(t,1,j-1)*k*B(t,2,j) - v(t,0,j)*h*B(t,1,j+1) + v(t,0,j-1)*h*(B(t,1,j) + B(t,1,j-1))
B(t,1,j) = B(t,1,j) + R*(A1/Ao - B(t,1,j))
         end if

         if ((u(t,0,j-1).gt.0) .and. (v(t,0,j-1).lt.0)) then
Ao       = 2*p(t,0,j-1)*h*k  + u(t,1,j-1)*k - v(t,0,j-1)*h
A1       = 2*e(t,0,j-1)*h*k - u(t,1,j-1)*k*B(t,2,j) + v(t,0,j-1)*h*B(t,1,j-1) - v(t,0,j)*h*(B(t,1,j) + B(t,1,j+1))
B(t,1,j) = B(t,1,j) + R*(A1/Ao - B(t,1,j))

         end if

         if ((u(t,0,j-1).lt.0) .and. (v(t,0,j-1).gt.0)) then
Ao       = 2*p(t,0,j-1)*h*k - u(t,0,j-1)*k + v(t,0,j)*h
A1       = 2*e(t,0,j-1)*h*k - v(t,0,j)*h*B(t,1,j+1) + u(t,0,j-1)*k*B(t,0,j) - u(t,1,j-1)*k*(B(t,1,j) + B(t,2,j)) + v(t,0,j-1)*h*(B(t,1,j-1) + B(t,1,j))
B(t,1,j) = B(t,1,j) + R*(A1/Ao - B(t,1,j))
B(t,0,j) = B(t,0,j) + R*(2*B(t,1,j) - B(t,2,j) - B(t,0,j))
         end if

         if  ((u(t,0,j-1).lt.0) .and. (v(t,0,j-1).lt.0)) then
Ao       = 2*p(t,0,j-1)*h*k - u(t,0,j-1)*k - v(t,0,j-1)*h
A1       = 2*e(t,0,j-1)*h*k + u(t,0,j-1)*k*B(t,0,j) + v(t,0,j-1)*h*B(t,1,j-1) - u(t,1,j-1)*k*(B(t,1,j) + B(t,2,j)) - v(t,0,j)*h*(B(t,1,j) + B(t,1,j+1))
B(t,1,j) = B(t,1,j) + R*(A1/Ao - B(t,1,j))
B(t,0,j) = B(t,0,j) + R*(2*B(t,1,j) - B(t,2,j) - B(t,0,j))
         end if

   end do
end do

                   ;***   boundary West
                   ;*     i = N-1

do t = 0,ntime-1
   do j = 1,nlat-1

         if ((u(t,N-2,j-1).gt.0) .and. (v(t,N-2,j-1).gt.0)) then
Ao       = 2*p(t,N-2,j-1)*h*k  + u(t,N-1,j-1)*k + v(t,N-2,j)*h
A1       = 2*h*k*e(t,N-2,j-1)  - u(t,N-1,j-1)*k*B(t,N,j) - v(t,N-2,j)*h*B(t,N-1,j+1) + u(t,N-2,j-1)*k*(B(t,N-1,j) + B(t,N-2,j)) + v(t,N-2,j-1)*h*(B(t,N-1,j) + B(t,N-1,j-1))
B(t,N-1,j) = B(t,N-1,j) + R*((A1/Ao) - B(t,N-1,j))
B(t,N,j) = B(t,N,j) + R*(2*B(t,N-1,j) - B(t,N-2,j) - B(t,N,j))

         end if

         if ((u(t,N-2,j-1).gt.0) .and. (v(t,N-2,j-1).lt.0)) then
Ao       = 2*p(t,N-2,j-1)*h*k + u(t,N-1,j-1)*k - v(t,N-2,j-1)*h
A1       = 2*e(t,N-2,j-1)*h*k - u(t,N-1,j-1)*k*B(t,N,j) + v(t,N-2,j-1)*h*B(t,N-1,j-1) + u(t,N-2,j-1)*k*(B(t,N-1,j) + B(t,N-2,j)) - v(t,N-2,j)*h*(B(t,N-1,j) + B(t,N-1,j+1))
B(t,N-1,j) = B(t,N-1,j) + R*(A1/Ao - B(t,N-1,j))
B(t,N,j) = B(t,N,j) + R*(2*B(t,N-1,j) - B(t,N-2,j) - B(t,N,j))
         end if

         if ((u(t,N-2,j-1).lt.0) .and. (v(t,N-2,j-1).gt.0)) then
Ao       = 2*p(t,N-2,j-1)*h*k - u(t,N-2,j-1)*k + v(t,N-2,j)*h
A1       = 2*e(t,N-2,j-1)*h*k - v(t,N-2,j)*h*B(t,N-1,j+1) + u(t,N-2,j-1)*k*B(t,N-2,j) + v(t,N-2,j-1)*h*(B(t,N-1,j-1) + B(t,N-1,j))
B(t,N-1,j) = B(t,N-1,j) + R*(A1/Ao - B(t,N-1,j))

         end if

         if  ((u(t,N-2,j-1).lt.0) .and. (v(t,N-2,j-1).lt.0)) then
Ao       = 2*p(t,N-2,j-1)*h*k - u(t,N-2,j-1)*k - v(t,N-2,j-1)*h
A1       = 2*e(t,N-2,j-1)*h*k + u(t,N-2,j-1)*k*B(t,N-2,j) + v(t,N-2,j-1)*h*B(t,N-1,j-1) - v(t,N-2,j)*h*(B(t,N-1,j) + B(t,N-1,j+1))
B(t,N-1,j) = B(t,N-1,j) + R*(A1/Ao - B(t,N-1,j))
         end if

   end do
end do

                   ;***   boundary South
                   ;*     j = 1
do t = 0,ntime-1
     do i = 1,nlon-1

         if ((u(t,i-1,0).gt.0) .and. (v(t,i-1,0).gt.0)) then
Ao       = 2*p(t,i-1,0)*h*k  + u(t,i,0)*k + v(t,i-1,1)*h
A1       = 2*h*k*e(t,i-1,0)  - u(t,i,0)*k*B(t,i+1,1) - v(t,i-1,1)*h*B(t,i,2) + u(t,i-1,0)*k*(B(t,i,1) + B(t,i-1,1))
C        = A1/Ao
B(t,i,1) = B(t,i,1) + R*((A1/Ao) - B(t,i,1))

         end if

         if ((u(t,i-1,0).lt.0) .and. (v(t,i-1,0).gt.0)) then
Ao       = 2*p(t,i-1,0)*h*k - u(t,i-1,0)*k + v(t,i-1,1)*h
A1       = 2*e(t,i-1,0)*h*k - v(t,i-1,1)*h*B(t,i,2) + u(t,i-1,0)*k*B(t,i-1,1) - u(t,i,0)*k*(B(t,i,1) + B(t,i+1,1))
B(t,i,1) = B(t,i,1) + R*(A1/Ao - B(t,i,1))
         end if


         if ((u(t,i-1,0).gt.0) .and. (v(t,i-1,0).lt.0)) then
Ao       = 2*p(t,i-1,0)*h*k  + u(t,i,0)*k - v(t,i-1,0)*h
A1       = 2*e(t,i-1,0)*h*k - u(t,i,0)*k*B(t,i+1,1) + v(t,i-1,0)*h*B(t,i,0) + u(t,i-1,0)*k*(B(t,i,1) + B(t,i-1,1)) - v(t,i-1,1)*h*(B(t,i,1) + B(t,i,2))
B(t,i,1) = B(t,i,1) + R*(A1/Ao - B(t,i,1))
B(t,i,0) = B(t,i,0) + R*(2*B(t,i,1) - B(t,i,2) - B(t,i,0))
         end if

         if  ((u(t,i-1,0).lt.0) .and. (v(t,i-1,0).lt.0)) then
Ao       = 2*p(t,i-1,0)*h*k - u(t,i-1,0)*k - v(t,i-1,0)*h
A1       = 2*e(t,i-1,0)*h*k + u(t,i-1,0)*k*B(t,i-1,1) + v(t,i-1,0)*h*B(t,i,0) - u(t,i,0)*k*(B(t,i,1) + B(t,i+1,1)) - v(t,i-1,1)*h*(B(t,i,1) + B(t,i,2))
B(t,i,1) = B(t,i,1) + R*(A1/Ao - B(t,i,1))
B(t,i,0) = B(t,i,0) + R*(2*B(t,i,1) - B(t,i,2) - B(t,i,0))
         end if

    end do
end do

                   ;***   boundary North
                   ;*     j = M-1
do t = 0,ntime-1
     do i = 1,nlon-1

         if ((u(t,i-1,M-2).gt.0) .and. (v(t,i-1,M-2).gt.0)) then
Ao       = 2*p(t,i-1,M-2)*h*k  + u(t,i,M-2)*k + v(t,i-1,M-1)*h
A1       = 2*h*k*e(t,i-1,M-2)  - u(t,i,M-2)*k*B(t,i+1,M-1) - v(t,i-1,M-1)*h*B(t,i,M) + u(t,i-1,M-2)*k*(B(t,i,M-1) + B(t,i-1,M-1)) + v(t,i-1,M-2)*h*(B(t,i,M-1) + B(t,i,M-2))
B(t,i,M-1) = B(t,i,M-1) + R*((A1/Ao) - B(t,i,M-1))
B(t,i,M) = B(t,i,M) + R*(2*B(t,i,M-1) - B(t,i,M-2) - B(t,i,M))
         end if

         if ((u(t,i-1,M-2).lt.0) .and. (v(t,i-1,M-2).gt.0)) then
Ao       = 2*p(t,i-1,M-2)*h*k - u(t,i-1,M-2)*k + v(t,i-1,M-1)*h
A1       = 2*e(t,i-1,M-2)*h*k - v(t,i-1,M-1)*h*B(t,i,M) + u(t,i-1,M-2)*k*B(t,i-1,M-1) - u(t,i,M-2)*k*(B(t,i,M-1) + B(t,i+1,M-1)) + v(t,i-1,M-2)*h*(B(t,i,M-2) + B(t,i,M-1))
B(t,i,M-1) = B(t,i,M-1) + R*(A1/Ao - B(t,i,M-1))
B(t,i,M) = B(t,i,M) + R*(2*B(t,i,M-1) - B(t,i,M-2) - B(t,i,M))
         end if

         if  ((u(t,i-1,M-2).lt.0) .and. (v(t,i-1,M-2).lt.0)) then
Ao       = 2*p(t,i-1,M-2)*h*k - u(t,i-1,M-2)*k - v(t,i-1,M-2)*h
A1       = 2*e(t,i-1,M-2)*h*k + u(t,i-1,M-2)*k*B(t,i-1,M-1) + v(t,i-1,M-2)*h*B(t,i,M-2) - u(t,i,M-2)*k*(B(t,i,M-1) + B(t,i+1,M-1))
B(t,i,M-1) = B(t,i,M-1) + R*(A1/Ao - B(t,i,M-1))
         end if

         if ((u(t,i-1,M-2).gt.0) .and. (v(t,i-1,M-2).lt.0)) then
Ao       = 2*p(t,i-1,M-2)*h*k  + u(t,i,M-2)*k - v(t,i-1,M-2)*h
A1       = 2*e(t,i-1,M-2)*h*k - u(t,i,M-2)*k*B(t,i+1,M-1) + v(t,i-1,M-2)*h*B(t,i,M-2) + u(t,i-1,M-2)*k*(B(t,i,M-1) + B(t,i-1,M-1))
B(t,i,M-1) = B(t,i,M-1) + R*(A1/Ao - B(t,i,M-1))
         end if

    end do
end do

do t = 0,ntime-1
   do i = 2,nlon-2
      do j = 2,nlat-2
                             ; Inner cells

         if ((u(t,i-1,j-1).gt.0) .and. (v(t,i-1,j-1).gt.0)) then
Ao       = 2*p(t,i-1,j-1)*h*k  + u(t,i,j-1)*k + v(t,i-1,j)*h
A1       = 2*h*k*e(t,i-1,j-1)  - u(t,i,j-1)*k*B(t,i+1,j) - v(t,i-1,j)*h*B(t,i,j+1) + u(t,i-1,j-1)*k*(B(t,i,j) + B(t,i-1,j)) + v(t,i-1,j-1)*h*(B(t,i,j) + B(t,i,j-1))
C        = A1/Ao
B(t,i,j) = B(t,i,j) + R*((A1/Ao) - B(t,i,j))

         end if

         if ((u(t,i-1,j-1).gt.0) .and. (v(t,i-1,j-1).lt.0)) then
Ao       = 2*p(t,i-1,j-1)*h*k  + u(t,i,j-1)*k - v(t,i-1,j-1)*h
A1       = 2*e(t,i-1,j-1)*h*k - u(t,i,j-1)*k*B(t,i+1,j) + v(t,i-1,j-1)*h*B(t,i,j-1) + u(t,i-1,j-1)*k*(B(t,i,j) + B(t,i-1,j)) - v(t,i-1,j)*h*(B(t,i,j) + B(t,i,j+1))
B(t,i,j) = B(t,i,j) + R*(A1/Ao - B(t,i,j))

         end if

         if ((u(t,i-1,j-1).lt.0) .and. (v(t,i-1,j-1).gt.0)) then
Ao       = 2*p(t,i-1,j-1)*h*k - u(t,i-1,j-1)*k + v(t,i-1,j)*h
A1       = 2*e(t,i-1,j-1)*h*k - v(t,i-1,j)*h*B(t,i,j+1) + u(t,i-1,j-1)*k*B(t,i-1,j) - u(t,i,j-1)*k*(B(t,i,j) + B(t,i+1,j)) + v(t,i-1,j-1)*h*(B(t,i,j-1) + B(t,i,j))
B(t,i,j) = B(t,i,j) + R*(A1/Ao - B(t,i,j))

         end if

         if  ((u(t,i-1,j-1).lt.0) .and. (v(t,i-1,j-1).lt.0)) then
Ao       = 2*p(t,i-1,j-1)*h*k - u(t,i-1,j-1)*k - v(t,i-1,j-1)*h
A1       = 2*e(t,i-1,j-1)*h*k + u(t,i-1,j-1)*k*B(t,i-1,j) + v(t,i-1,j-1)*h*B(t,i,j-1) - u(t,i,j-1)*k*(B(t,i,j) + B(t,i+1,j)) - v(t,i-1,j)*h*(B(t,i,j) + B(t,i,j+1))
B(t,i,j) = B(t,i,j) + R*(A1/Ao - B(t,i,j))
         end if

        end do
     end do
  end do

K = K+1

end do

latitude  = Fx&lat(0:nlat-2)
longitude = Fx&lon(0:nlon-2)
copy_VarCoords(Fx(time|:,lon|0:nlon-2,lat|0:nlat-2),B(:,1:nlon-1,1:nlat-1))

rho = B(:,1:nlon-1,1:nlat-1)

;-------- 337
;rho(lon|:,lat|:,time|337) = (1./2.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|1:325:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|349::12)))
;-------- 348
;rho(lon|:,lat|:,time|348) = (1./2.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|0:336:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|360::12)))
;-------- 351
;rho(lon|:,lat|:,time|351) = (1./2.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|3:339:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|363::12)))
;-------- 124
;rho(lon|:,lat|:,time|124) = (1./2.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|4:112:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|136::12)))

;-------- 246
;rho(lon|:,lat|:,time|66)  = (1./5.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|6:54:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|78:90:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|126:234:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|25-76:-504:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|378::12)))
;rho(lon|:,lat|:,time|102) = (1./5.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|6:54:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|78:90:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|126:234:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|25-76:-504:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|378::12)))
;rho(lon|:,lat|:,time|114) = (1./5.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|6:54:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|78:90:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|126:234:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|25-76:-504:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|378::12)))
;rho(lon|:,lat|:,time|246) = (1./5.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|6:54:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|78:90:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|126:234:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|25-76:-504:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|378::12)))
;rho(lon|:,lat|:,time|366) = (1./5.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|6:54:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|78:90:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|126:234:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|25-76:-504:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|378::12)))

;-------- 247
;rho(lon|:,lat|:,time|91)  = (1./8.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|7:79:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|115:139:12)) + rho(lon|:,lat|:,time|163) + dim_avg_Wrap(rho(lon|:,lat|:,time|187:283:12)) + rho(lon|:,lat|:,time|307) + dim_avg_Wrap(rho(lon|:,lat|:,time|331:355:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|379:391:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|415::12)))
;rho(lon|:,lat|:,time|103) = (1./8.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|7:79:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|115:139:12)) + rho(lon|:,lat|:,time|163) + dim_avg_Wrap(rho(lon|:,lat|:,time|187:283:12)) + rho(lon|:,lat|:,time|307) + dim_avg_Wrap(rho(lon|:,lat|:,time|331:355:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|379:391:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|415::12)))
;rho(lon|:,lat|:,time|151) = (1./8.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|7:79:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|115:139:12)) + rho(lon|:,lat|:,time|163) + dim_avg_Wrap(rho(lon|:,lat|:,time|187:283:12)) + rho(lon|:,lat|:,time|307) + dim_avg_Wrap(rho(lon|:,lat|:,time|331:355:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|379:391:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|415::12)))
;rho(lon|:,lat|:,time|175) = (1./8.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|7:79:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|115:139:12)) + rho(lon|:,lat|:,time|163) + dim_avg_Wrap(rho(lon|:,lat|:,time|187:283:12)) + rho(lon|:,lat|:,time|307) + dim_avg_Wrap(rho(lon|:,lat|:,time|331:355:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|379:391:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|415::12)))
;rho(lon|:,lat|:,time|295) = (1./8.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|7:79:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|115:139:12)) + rho(lon|:,lat|:,time|163) + dim_avg_Wrap(rho(lon|:,lat|:,time|187:283:12)) + rho(lon|:,lat|:,time|307) + dim_avg_Wrap(rho(lon|:,lat|:,time|331:355:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|379:391:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|415::12)))
;rho(lon|:,lat|:,time|319) = (1./8.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|7:79:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|115:139:12)) + rho(lon|:,lat|:,time|163) + dim_avg_Wrap(rho(lon|:,lat|:,time|187:283:12)) + rho(lon|:,lat|:,time|307) + dim_avg_Wrap(rho(lon|:,lat|:,time|331:355:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|379:391:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|415::12)))
;rho(lon|:,lat|:,time|367) = (1./8.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|7:79:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|115:139:12)) + rho(lon|:,lat|:,time|163) + dim_avg_Wrap(rho(lon|:,lat|:,time|187:283:12)) + rho(lon|:,lat|:,time|307) + dim_avg_Wrap(rho(lon|:,lat|:,time|331:355:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|379:391:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|415::12)))
;rho(lon|:,lat|:,time|403) = (1./8.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|7:79:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|115:139:12)) + rho(lon|:,lat|:,time|163) + dim_avg_Wrap(rho(lon|:,lat|:,time|187:283:12)) + rho(lon|:,lat|:,time|307) + dim_avg_Wrap(rho(lon|:,lat|:,time|331:355:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|379:391:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|415::12)))

;-------- 248
;rho(lon|:,lat|:,time|68)  = (1./6.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|8:56:12)) + rho(lon|:,lat|:,time|80) + dim_avg_Wrap(rho(lon|:,lat|:,time|104:147:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|171:164:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|188:200:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|224::12)))
;rho(lon|:,lat|:,time|92)  = (1./6.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|8:56:12)) + rho(lon|:,lat|:,time|80) + dim_avg_Wrap(rho(lon|:,lat|:,time|104:147:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|171:164:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|188:200:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|224::12)))
;rho(lon|:,lat|:,time|159) = (1./6.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|8:56:12)) +rho(lon|:,lat|:,time|80) + dim_avg_Wrap(rho(lon|:,lat|:,time|104:147:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|171:164:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|188:200:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|224::12)))
;rho(lon|:,lat|:,time|176) = (1./6.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|8:56:12)) +rho(lon|:,lat|:,time|80) + dim_avg_Wrap(rho(lon|:,lat|:,time|104:147:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|171:164:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|188:200:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|224::12)))
;rho(lon|:,lat|:,time|212) = (1./6.)*( dim_avg_Wrap(rho(lon|:,lat|:,time|8:56:12)) +rho(lon|:,lat|:,time|80) + dim_avg_Wrap(rho(lon|:,lat|:,time|104:147:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|171:164:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|188:200:12)) + dim_avg_Wrap(rho(lon|:,lat|:,time|224::12)))

;print(ut_calendar(rho&time(0:395),1))

       time_avgB       = B(lat|1:nlat-1,lon|1:nlon-1,time|0)
       time_avgB       = dim_avg(B(lat|1:nlat-1,lon|1:nlon-1,time|:))
       time_avgB@units = " "
                              ;------     allocate coordinates
     do i = 0,nlat-2
       time_avgB&lat(i)   = latitude(i) + 1.25
     end do
     do i = 0,nlon-2
       time_avgB&lon(i)   = longitude(i) + 1.25
     end do

     do i = 0,nlat-2
       rho&lat(i)   = latitude(i) + 1.25
     end do
     do i = 0,nlon-2
       rho&lon(i)   = longitude(i) + 1.25
     end do

     do i = 0,nlat-2
       e&lat(i)   = latitude(i) + 1.25
     end do
     do i = 0,nlon-2
       e&lon(i)   = longitude(i)+ 1.25
     end do

     do i = 0,nlat-2
       p&lat(i)   = latitude(i) + 1.25
     end do
     do i = 0,nlon-2
       p&lon(i)   = longitude(i)+ 1.25
     end do

       time_avgB       = dim_avg_Wrap(rho(lat|:,lon|:,time|0::12))

rho@units       = "percent"
e@units         = "m/s multiply by 86400*30*1e3 convert in mm/month"
p@units         = "m/s multiply by 86400*30*1e3 convert in mm/month"

rho@long_name   = "precipitation recycling"
e@long_name     = "evaporation"
p@long_name     = "precipitation"

recyl = addfile("data/recycl_ncep_new_500.nc","c")   ;----    save computed variables in a NetCDF file
recyl->rec = rho
recyl->evp = e
recyl->pre = p

print("minlon = "+time_avgB&lon(0)+"  maxlon = "+time_avgB&lon(nlon-2)+"   minlat = "+time_avgB&lat(0)+"   maxlat = "+time_avgB&lat(nlat-2))
print("minlon = "+rho&lon(0)+"        maxlon = "+rho&lon(nlon-2)+"         minlat = "+rho&lat(0)+"         maxlat = "+rho&lat(nlat-2))
print("minlon = "+Fx&lon(0)+"         maxlon = "+Fx&lon(nlon-1)+"          minlat = "+Fx&lat(0)+"          maxlat = "+Fx&lat(nlat-1))

end
