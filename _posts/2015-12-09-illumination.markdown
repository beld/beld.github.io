---
layout:     post
title:      "Illumination"
subtitle:   ""
date:       2015-12-09 23:30:00
author:     "Beld"
header-img: "img/post-bg-cv.jpg"
tags:
    - Computer Vision
---


**Illumination**: The amount of light a patch receives depends on the overall intensity of the light, and on the geometry.

Reflectance Models:
Description of how light energy incident on an object is transferred from the object to the camera sensor.

Diffuse reflection: an incident ray is reflected at many angles rather than at just one angle as in the case of specular reflection.
Lambertian reflectance: ideal diffusely reflecting surface. The apparent brightness of a Lambertian surface to an observer is the same regardless of the observer's angle of view.
Lambert's cosine law: an ideal diffusely reflecting surface is directly proportional to the cosine of the angle θ between the direction of the incident light and the surface normal.

Radiometry: how "bright" will surfaces be?
What is "brightness"? - measuring light interactions between light and surfaces.

Solid Angle [立体角] (analogy with angle, in radians): subtended by a patch is the area covered by the patch on a *unit* sphere.

Radiance ［辐射］:
- Measure the "amount of light" at a point, in a direction
- Definition: *Radiant power per unit foreshortened area per unit solid angle*
- 强翻：每单位立体角每单位透视减少面积的辐射功率。
- 吐槽1: 看了定义更不懂了。
- 吐槽2: 想维基中foreshortened对应的翻译，结果发现perspective projection，perspective，foreshortening统统跳到了只有几行的透视投影页面。


Irradiance ［辐照］:
- Measure How much light is arriving at a surface?
- a function of incoming angle
- Definition: *Incident power per unit area not foreshortened*
- Crucial property: Total power arriving at the surface is given by adding irradiance over all incoming angles.

surfaces:
- Many effects when a light strikes a surface: absorbed, transmitted, reflected, scattered
- Assume that
  - surfaces don’t fluoresce
  - surfaces don’t emit light (i.e. are cool)
  - all the light leaving a point is due to that arriving at that point

BRDF (Bidirectional Reflectance Distribution Function) [双向反射分布函数]:
- Definition: the ratio of the radiance in the outgoing direction to the incident irradiance
*用来定义给定入射方向上的辐射照度（irradiance）如何影响给定出射方向上的辐射率（radiance）。更笼统地说，它描述了入射光线经过某个表面反射后如何在各个出射方向上分布——这可以是从理想镜面反射到漫反射、各向同性（isotropic）或者各向异性（anisotropic）的各种反射。*

Albedo or *reflection coefficient*
In general, the albedo depends on the directional distribution of incident radiation, except for Lambertian surfaces, which scatter radiation in all directions according to a cosine function and therefore have an albedo that is independent of the incident distribution. In practice, a bidirectional reflectance distribution function (BRDF) may be required to accurately characterize the scattering properties of a surface, but albedo is very useful as a first approximation.



Specular surfaces
- radiation arriving along a direction leaves along the specular direction
- reflect about normal
- some fraction is absorbed, some reflected
– on real surfaces, energy usually goes into a lobe of directions
– can write a BRDF, but requires the use of funny functions


Color

Color receptors

METAMERISM: Two different Spectral Energy Distributions with the same RED, GREEN, BLUE response are termed metamers.

Three-CCD camera
single chip camera
