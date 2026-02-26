ThisBuild / tlBaseVersion := "0.15"

ThisBuild / organization     := "ai.entrolution"
ThisBuild / organizationName := "Greg von Nessi"
ThisBuild / startYear        := Some(2023)
ThisBuild / licenses         := Seq(License.Apache2)
ThisBuild / developers ++= List(
  tlGitHubDev("gvonness", "Greg von Nessi")
)

// CI configuration - Java 21 required for Smile 3.x
ThisBuild / githubWorkflowJavaVersions := Seq(JavaSpec.temurin("21"))

scalaVersion                    := DependencyVersions.scala2p13Version
ThisBuild / crossScalaVersions  := Seq(DependencyVersions.scala2p13Version, DependencyVersions.scala3Version)
ThisBuild / tlVersionIntroduced := Map("2.13" -> "0.15", "3" -> "0.15")

Global / idePackagePrefix := Some("ai.entrolution")
Global / excludeLintKeys += idePackagePrefix

lazy val commonSettings = Seq(
  scalaVersion := DependencyVersions.scala2p13Version,
  scalacOptions ++= (
    if (scalaVersion.value.startsWith("2."))
      Seq("-Xlint:_", "-Ywarn-unused:-implicits", "-Ywarn-value-discard", "-Ywarn-dead-code")
    else
      Seq("-Wunused:all")
  )
)

lazy val thylacine = (project in file("."))
  .settings(
    commonSettings,
    name := "thylacine",
    libraryDependencies ++= Dependencies.thylacine,
    crossScalaVersions := Seq(
      DependencyVersions.scala2p13Version,
      DependencyVersions.scala3Version
    ),
    Test / parallelExecution := false
  )
